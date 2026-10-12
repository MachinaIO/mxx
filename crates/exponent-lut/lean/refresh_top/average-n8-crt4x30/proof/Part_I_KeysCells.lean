import TopInst
import TopKeys
import Part_I_Cells
import ExponentFunctional

/-! The keys of the online PRF cells (written by `generator/gen_keys_cells.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

/-- `rns-bgg/input/WireId(1)/0`, the hash tag of the label's public matrices. -/
def labelPrefix : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87,
  105, 114, 101, 73, 100, 40, 49, 41, 47, 48]

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `0` of the mask PRF (online `115`) are those the
preprocessing computes (`35`). -/
theorem keysCell115 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q7 N 1 14} {apay : Fin 5 → ExactMatrix Q7 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : ExactMatrix Q7 N 1 12} {bot0 bot1 : ExactMatrix Q6 N 1 10}
    {outs : Blocks Q6 10}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_115 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q7 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_35 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q7 N 1 14 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_115 at hon
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_6_0, hon⟩ := hon
  obtain ⟨w_7_0, hon⟩ := hon
  obtain ⟨w_7_concat_1, hon⟩ := hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  obtain ⟨w_14_concat_1, hon⟩ := hon
  obtain ⟨w_15_0, hon⟩ := hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_19_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hon
  obtain ⟨w_28_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  obtain ⟨w_30_0, hon⟩ := hon
  obtain ⟨w_30_concat_1, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  obtain ⟨w_34_0, hon⟩ := hon
  obtain ⟨w_35_0, hon⟩ := hon
  extract_lets -merge w_36_0 at hon
  obtain ⟨w_37_0, hon⟩ := hon
  extract_lets -merge w_40_0 at hon
  obtain ⟨w_41_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_43_concat_1, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_46_0, hon⟩ := hon
  obtain ⟨w_48_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_50_0, hon⟩ := hon
  extract_lets -merge w_51_0 at hon
  obtain ⟨w_52_0, hon⟩ := hon
  obtain ⟨w_53_0, hon⟩ := hon
  obtain ⟨w_54_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_58_0, hon⟩ := hon
  extract_lets -merge w_59_0 at hon
  obtain ⟨w_60_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  obtain ⟨w_65_0, hon⟩ := hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_67_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  obtain ⟨w_69_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_115.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_115.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_115.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_115.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_115.constraints_4] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, -, -, -, -, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, -, -, -, -, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_35 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_4_0 pw_10_0 pw_21_0 pw_24_0 pw_31_0 pw_38_0 pw_40_0 pw_58_0 pw_36_0 pw_39_0 pw_41_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_2_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_5_0, hpre⟩ := hpre
  extract_lets -merge pw_6_0 at hpre
  obtain ⟨pw_7_0, hpre⟩ := hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  extract_lets -merge pw_12_0 pw_13_0 at hpre
  obtain ⟨pw_14_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  obtain ⟨pw_17_0, hpre⟩ := hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  obtain ⟨pw_19_0, hpre⟩ := hpre
  extract_lets -merge pw_20_0 at hpre
  obtain ⟨pw_22_0, hpre⟩ := hpre
  extract_lets -merge pw_23_0 at hpre
  obtain ⟨pw_25_0, hpre⟩ := hpre
  extract_lets -merge pw_26_0 pw_27_0 at hpre
  obtain ⟨pw_28_0, hpre⟩ := hpre
  obtain ⟨pw_29_0, hpre⟩ := hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 at hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  extract_lets -merge pw_35_0 pw_37_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  extract_lets -merge pw_43_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  extract_lets -merge pw_47_0 at hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_49_0, hpre⟩ := hpre
  extract_lets -merge pw_50_0 pw_51_0 at hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  obtain ⟨pw_53_0, hpre⟩ := hpre
  obtain ⟨pw_54_0, hpre⟩ := hpre
  extract_lets -merge pw_63_0 at hpre
  obtain ⟨pw_55_0, hpre⟩ := hpre
  obtain ⟨pw_56_0, hpre⟩ := hpre
  extract_lets -merge pw_57_0 at hpre
  obtain ⟨pw_59_0, hpre⟩ := hpre
  extract_lets -merge pw_60_0 pw_61_0 pw_62_0 pw_64_0 at hpre
  obtain ⟨pw_65_0, hpre⟩ := hpre
  extract_lets -merge pw_66_0 at hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  obtain ⟨pw_68_0, hpre⟩ := hpre
  extract_lets -merge pw_69_0 pw_70_0 at hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  obtain ⟨pw_75_0, hpre⟩ := hpre
  obtain ⟨pw_76_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_35.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_35.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_35.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_35.constraints_3] at hpre
  obtain ⟨-, -, -, ph1_0, -, -, -, ph2_0, ph3_0, ph5_0, -, -, -, ph7_0, -, -, -, ph8_0, ph9_0, ph11_0, -, -, -, -, -, -, ph14_0, -, -, -, -, -, -, ph15_0, ph16_0, -, -, -, -, ph17_0, ph18_0, ph19_0, ph22_0, ph25_0, -, -, -, -, -, -, ph28_0, -, -, -, -, -, -, ph29_0, ph30_0, ph32_0, ph33_0, ph34_0, ph42_0, ph46_0, ph48_0, ph49_0, -, -, -, -, -, -, ph52_0, ph53_0, -, -, -, -, ph54_0, ph55_0, ph56_0, ph59_0, ph65_0, ph67_0, ph68_0, -, -, -, -, -, -, ph71_0, ph72_0, -, -, -, -, -, -, ph73_0, ph74_0, -, -, -, -, -, -, ph75_0, ph76_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h4_0 h6_0 h11_0 h13_0
  obtain ⟨hk, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_4_0 := by
    rw [hashSample_eq ph1_0, hw4, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e2 : pw_2_0 = w_6_0 := by
    rw [hashSample_eq ph2_0, hw6, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e7 : pw_7_0 = w_11_0 := by
    rw [hashSample_eq ph7_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e8 : pw_8_0 = w_13_0 := by
    rw [hashSample_eq ph8_0, hw13, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The public multiplication by the one encoding.
  obtain ⟨t20, b20⟩ := decompose_stack Backend.layout8 layoutQ7 rfl h18_0 h19_0 ph5_0 ph11_0
  have r20 := rebase_self (by inst_num) h20_0
  obtain ⟨t35, b35⟩ := decompose_stack Backend.layout8 layoutQ7 rfl h33_0 h34_0 ph22_0 ph25_0
  have r35 := rebase_self (by inst_num) h35_0
  obtain ⟨-, s21, s21'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_20_0
  obtain ⟨-, s36, s36'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_35_0
  obtain ⟨p3a, p3b⟩ := stack2_mul_rows ph3_0 pw_5_0
  obtain ⟨p9a, p9b⟩ := stack2_mul_rows ph9_0 pw_11_0
  obtain ⟨p3a', p3b'⟩ := stack2_mul_rows ph3_0 pw_22_0
  obtain ⟨p9a', p9b'⟩ := stack2_mul_rows ph9_0 pw_25_0
  have k28 : w_28_0 = pw_14_0 := by
    rw [slice_rowOf (by inst_num) h28_0, slice_rowOf (by inst_num) ph14_0]
    show rowOf (w_15_0 * w_20_0) 1 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 0
    rw [s21, pairMul, rowOf_add, p3a, p9a, r20, t20, b20, e1, e7]
    rfl
  have k29 : w_29_0 = pw_15_0 := by
    rw [slice_rowOf (by inst_num) h29_0, slice_rowOf (by inst_num) ph15_0]
    show rowOf (w_15_0 * w_20_0) 2 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 1
    rw [s21', pairMul, rowOf_add, p3b, p9b, r20, t20, b20, e2, e8]
    rfl
  have k41 : w_41_0 = pw_28_0 := by
    rw [slice_rowOf (by inst_num) h41_0, slice_rowOf (by inst_num) ph28_0]
    show rowOf (w_15_0 * w_35_0) 1 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 0
    rw [s36, pairMul, rowOf_add, p3a', p9a', r35, t35, b35, e1, e7]
    rfl
  have k42 : w_42_0 = pw_29_0 := by
    rw [slice_rowOf (by inst_num) h42_0, slice_rowOf (by inst_num) ph29_0]
    show rowOf (w_15_0 * w_35_0) 2 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 1
    rw [s36', pairMul, rowOf_add, p3b', p9b', r35, t35, b35, e2, e8]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_46_0 = pw_17_0 := (hashSample_eq h46_0).trans (hashSample_eq ph17_0).symm
  have et1 : w_54_0 = pw_54_0 := (hashSample_eq h54_0).trans (hashSample_eq ph54_0).symm
  have eb0 : w_47_0 = pw_47_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_55_0 = pw_66_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h48_0
  rw [et1, eb1] at h56_0
  obtain ⟨t50, b50⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h48_0 h49_0 ph18_0 ph48_0
  have t50' := rebase_topRows h50_0 ph19_0 t50
  have b50' := rebase_bottomRows h50_0 ph49_0 b50
  obtain ⟨t58, b58⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h56_0 h57_0 ph55_0 ph67_0
  have t58' := rebase_topRows h58_0 ph56_0 t58
  have b58' := rebase_bottomRows h58_0 ph68_0 b58
  obtain ⟨-, s51, s51'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_50_0
  obtain ⟨-, s59, s59'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_58_0
  obtain ⟨q16a, q16b⟩ := stack2_mul_rows ph16_0 pw_19_0
  obtain ⟨q30a, q30b⟩ := stack2_mul_rows ph30_0 pw_49_0
  obtain ⟨q16a', q16b'⟩ := stack2_mul_rows ph16_0 pw_56_0
  obtain ⟨q30a', q30b'⟩ := stack2_mul_rows ph30_0 pw_68_0
  have k52 : w_52_0 = pw_52_0 := by
    rw [slice_rowOf (by inst_num) h52_0, slice_rowOf (by inst_num) ph52_0]
    show rowOf (w_44_0 * w_50_0) 1 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 0
    rw [s51, pairMul, rowOf_add, q16a, q30a, t50', b50', k28, k41]
    rfl
  have k62 : w_62_0 = pw_73_0 := by
    rw [slice_rowOf (by inst_num) h62_0, slice_rowOf (by inst_num) ph73_0]
    show rowOf (w_44_0 * w_50_0) 2 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 1
    rw [s51', pairMul, rowOf_add, q16b, q30b, t50', b50', k29, k42]
    rfl
  have k60 : w_60_0 = pw_71_0 := by
    rw [slice_rowOf (by inst_num) h60_0, slice_rowOf (by inst_num) ph71_0]
    show rowOf (w_44_0 * w_58_0) 1 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 0
    rw [s59, pairMul, rowOf_add, q16a', q30a', t58', b58', k28, k41]
    rfl
  have k64 : w_64_0 = pw_75_0 := by
    rw [slice_rowOf (by inst_num) h64_0, slice_rowOf (by inst_num) ph75_0]
    show rowOf (w_44_0 * w_58_0) 2 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 1
    rw [s59', pairMul, rowOf_add, q16b', q30b', t58', b58', k29, k42]
    rfl
  rw [k52] at h53_0
  rw [k62] at h63_0
  rw [k60] at h61_0
  rw [k64] at h65_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h53_0 ph53_0, blockModSwitchRuns_unique h61_0 ph72_0,
    blockModSwitchRuns_unique h63_0 ph74_0, blockModSwitchRuns_unique h65_0 ph76_0⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `1` of the mask PRF (online `119`) are those the
preprocessing computes (`39`). -/
theorem keysCell119 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q7 N 1 14} {apay : Fin 5 → ExactMatrix Q7 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : Fin 8 → ExactMatrix Q6 N 1 10}
    {bot0 bot1 : ExactMatrix Q5 N 1 8} {outs : Blocks Q5 8}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_119 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q7 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_39 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q7 N 1 14 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_119 at hon
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_5_0, hon⟩ := hon
  extract_lets -merge w_6_0 at hon
  obtain ⟨w_7_0, hon⟩ := hon
  extract_lets -merge w_8_0 at hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  extract_lets -merge w_15_0 at hon
  obtain ⟨w_16_0, hon⟩ := hon
  extract_lets -merge w_17_0 at hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_23_0, hon⟩ := hon
  obtain ⟨w_23_concat_1, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 at hon
  obtain ⟨w_27_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  extract_lets -merge w_30_0 at hon
  obtain ⟨w_31_0, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  extract_lets -merge w_34_0 at hon
  obtain ⟨w_35_0, hon⟩ := hon
  obtain ⟨w_36_0, hon⟩ := hon
  obtain ⟨w_36_concat_1, hon⟩ := hon
  obtain ⟨w_37_0, hon⟩ := hon
  obtain ⟨w_40_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_45_0, hon⟩ := hon
  extract_lets -merge w_46_0 at hon
  obtain ⟨w_47_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_51_0, hon⟩ := hon
  obtain ⟨w_52_0, hon⟩ := hon
  extract_lets -merge w_53_0 w_54_0 at hon
  obtain ⟨w_55_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_57_concat_1, hon⟩ := hon
  obtain ⟨w_59_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  extract_lets -merge w_65_0 at hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  extract_lets -merge w_69_0 w_70_0 at hon
  obtain ⟨w_71_0, hon⟩ := hon
  obtain ⟨w_72_0, hon⟩ := hon
  obtain ⟨w_73_0, hon⟩ := hon
  obtain ⟨w_73_concat_1, hon⟩ := hon
  obtain ⟨w_74_0, hon⟩ := hon
  obtain ⟨w_76_0, hon⟩ := hon
  obtain ⟨w_78_0, hon⟩ := hon
  obtain ⟨w_79_0, hon⟩ := hon
  obtain ⟨w_80_0, hon⟩ := hon
  extract_lets -merge w_81_0 at hon
  obtain ⟨w_82_0, hon⟩ := hon
  obtain ⟨w_83_0, hon⟩ := hon
  obtain ⟨w_84_0, hon⟩ := hon
  obtain ⟨w_86_0, hon⟩ := hon
  obtain ⟨w_87_0, hon⟩ := hon
  obtain ⟨w_88_0, hon⟩ := hon
  extract_lets -merge w_89_0 at hon
  obtain ⟨w_90_0, hon⟩ := hon
  obtain ⟨w_91_0, hon⟩ := hon
  obtain ⟨w_92_0, hon⟩ := hon
  obtain ⟨w_93_0, hon⟩ := hon
  obtain ⟨w_94_0, hon⟩ := hon
  obtain ⟨w_95_0, hon⟩ := hon
  obtain ⟨w_96_0, hon⟩ := hon
  obtain ⟨w_97_0, hon⟩ := hon
  obtain ⟨w_98_0, hon⟩ := hon
  obtain ⟨w_99_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_119.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_119.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_119.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_119.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_119.constraints_4, W_Stage_online_eval.parallel_parallel_generatedRoot_290_119.constraints_5] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, -, -, -, -, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, -, -, -, -, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_39 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_13_0 pw_25_0 pw_37_0 pw_41_0 pw_49_0 pw_56_0 pw_58_0 pw_76_0 pw_2_0 pw_14_0 pw_54_0 pw_57_0 pw_59_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_4_0, hpre⟩ := hpre
  extract_lets -merge pw_5_0 at hpre
  obtain ⟨pw_6_0, hpre⟩ := hpre
  extract_lets -merge pw_7_0 at hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  extract_lets -merge pw_10_0 at hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  obtain ⟨pw_12_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  extract_lets -merge pw_17_0 at hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  extract_lets -merge pw_19_0 at hpre
  obtain ⟨pw_20_0, hpre⟩ := hpre
  obtain ⟨pw_21_0, hpre⟩ := hpre
  extract_lets -merge pw_22_0 at hpre
  obtain ⟨pw_23_0, hpre⟩ := hpre
  obtain ⟨pw_24_0, hpre⟩ := hpre
  obtain ⟨pw_26_0, hpre⟩ := hpre
  obtain ⟨pw_27_0, hpre⟩ := hpre
  extract_lets -merge pw_28_0 pw_29_0 at hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_31_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  obtain ⟨pw_35_0, hpre⟩ := hpre
  extract_lets -merge pw_36_0 at hpre
  obtain ⟨pw_38_0, hpre⟩ := hpre
  obtain ⟨pw_39_0, hpre⟩ := hpre
  extract_lets -merge pw_40_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  obtain ⟨pw_43_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  obtain ⟨pw_47_0, hpre⟩ := hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_50_0, hpre⟩ := hpre
  extract_lets -merge pw_62_0 at hpre
  obtain ⟨pw_51_0, hpre⟩ := hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  extract_lets -merge pw_53_0 pw_55_0 at hpre
  obtain ⟨pw_60_0, hpre⟩ := hpre
  extract_lets -merge pw_61_0 pw_63_0 at hpre
  obtain ⟨pw_64_0, hpre⟩ := hpre
  extract_lets -merge pw_65_0 at hpre
  obtain ⟨pw_66_0, hpre⟩ := hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  extract_lets -merge pw_68_0 pw_69_0 at hpre
  obtain ⟨pw_70_0, hpre⟩ := hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  extract_lets -merge pw_81_0 at hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  extract_lets -merge pw_75_0 at hpre
  obtain ⟨pw_77_0, hpre⟩ := hpre
  extract_lets -merge pw_78_0 pw_79_0 pw_80_0 pw_82_0 at hpre
  obtain ⟨pw_83_0, hpre⟩ := hpre
  extract_lets -merge pw_84_0 at hpre
  obtain ⟨pw_85_0, hpre⟩ := hpre
  obtain ⟨pw_86_0, hpre⟩ := hpre
  extract_lets -merge pw_87_0 pw_88_0 at hpre
  obtain ⟨pw_89_0, hpre⟩ := hpre
  obtain ⟨pw_90_0, hpre⟩ := hpre
  obtain ⟨pw_91_0, hpre⟩ := hpre
  obtain ⟨pw_92_0, hpre⟩ := hpre
  obtain ⟨pw_93_0, hpre⟩ := hpre
  obtain ⟨pw_94_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_39.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_39.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_39.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_39.constraints_3, W_Stage_preprocess.parallel_parallel_generatedRoot_407_39.constraints_4] at hpre
  obtain ⟨-, -, -, ph1_0, ph3_0, ph4_0, ph6_0, ph8_0, -, -, -, ph9_0, ph11_0, ph12_0, -, ph14_0, ph14_0_1, ph15_0, ph16_0, -, -, -, ph18_0, ph20_0, -, -, -, ph21_0, ph23_0, ph24_0, ph14_0_2, ph14_0_3, ph26_0, ph27_0, -, -, -, -, -, -, ph30_0, -, -, -, -, -, -, ph31_0, ph32_0, -, -, -, -, ph33_0, ph34_0, ph35_0, ph14_0_4, ph14_0_5, ph38_0, ph39_0, ph14_0_6, ph14_0_7, ph42_0, ph43_0, -, -, -, -, -, -, ph46_0, -, -, -, -, -, -, ph47_0, ph48_0, ph50_0, ph51_0, ph52_0, ph60_0, ph64_0, ph66_0, ph67_0, -, -, -, -, -, -, ph70_0, ph71_0, -, -, -, -, ph72_0, ph73_0, ph74_0, ph77_0, ph83_0, ph85_0, ph86_0, -, -, -, -, -, -, ph89_0, ph90_0, -, -, -, -, -, -, ph91_0, ph92_0, -, -, -, -, -, -, ph93_0, ph94_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h11_0 h20_0 h29_0 h33_0
  obtain ⟨hk, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_11_0 := by
    rw [hashSample_eq ph1_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e9 : pw_9_0 = w_20_0 := by
    rw [hashSample_eq ph9_0, hw20, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e18 : pw_18_0 = w_29_0 := by
    rw [hashSample_eq ph18_0, hw29, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e21 : pw_21_0 = w_33_0 := by
    rw [hashSample_eq ph21_0, hw33, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The alignment `D(p G)` of the label.
  have g13 : w_13_0 = pw_3_0 := gadgetMatrixRuns_unique h13_0 ph3_0
  rw [g13] at h14_0
  have g14 : w_14_0 = pw_4_0 := centeredRebaseRuns_unique h14_0 ph4_0
  have g16 : w_16_0 = pw_6_0 := by
    apply gadgetDecomposeRuns_unique h16_0
    show gadgetDecomposeRuns _ _ _ (matrixMulScalarLeft _ w_14_0) _
    rw [g14]
    exact ph6_0
  have a18 : w_18_0 = pw_8_0 := by
    apply blockModSwitchRuns_unique h18_0
    show blockModSwitchRuns _ _ (w_11_0 * w_16_0) _
    rw [← e1, g16]
    exact ph8_0
  have a22 : w_22_0 = pw_11_0 := by
    apply blockModSwitchRuns_unique h22_0
    show blockModSwitchRuns _ _ (w_20_0 * w_16_0) _
    rw [← e9, g16]
    exact ph11_0
  have a31 : w_31_0 = pw_20_0 := by
    apply blockModSwitchRuns_unique h31_0
    show blockModSwitchRuns _ _ (w_29_0 * w_16_0) _
    rw [← e18, g16]
    exact ph20_0
  have a35 : w_35_0 = pw_23_0 := by
    apply blockModSwitchRuns_unique h35_0
    show blockModSwitchRuns _ _ (w_33_0 * w_16_0) _
    rw [← e21, g16]
    exact ph23_0
  -- The members of bucket `0`.
  have ho : w_39_0 = pw_14_0 := rfl
  rw [ho] at h40_0 h42_0 h59_0 h61_0
  have m40 : w_40_0 = pw_15_0 := familyGetDynamic_unique h40_0 ph15_0
  have m42 : w_42_0 = pw_26_0 := familyGetDynamic_unique h42_0 ph26_0
  have m59 : w_59_0 = pw_38_0 := familyGetDynamic_unique h59_0 ph38_0
  have m61 : w_61_0 = pw_42_0 := familyGetDynamic_unique h61_0 ph42_0
  rw [m40, m42] at h43_0
  rw [m59, m61] at h62_0
  obtain ⟨t45, b45⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h43_0 h44_0 ph16_0 ph27_0
  have r45 := rebase_self (by inst_num) h45_0
  obtain ⟨t64, b64⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h62_0 h63_0 ph39_0 ph43_0
  have r64 := rebase_self (by inst_num) h64_0
  obtain ⟨-, s46, s46'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_45_0
  obtain ⟨-, s65, s65'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_64_0
  obtain ⟨p12a, p12b⟩ := stack2_mul_rows ph12_0 pw_16_0
  obtain ⟨p24a, p24b⟩ := stack2_mul_rows ph24_0 pw_27_0
  obtain ⟨p12a', p12b'⟩ := stack2_mul_rows ph12_0 pw_39_0
  obtain ⟨p24a', p24b'⟩ := stack2_mul_rows ph24_0 pw_43_0
  have k55 : w_55_0 = pw_30_0 := by
    rw [slice_rowOf (by inst_num) h55_0, slice_rowOf (by inst_num) ph30_0]
    show rowOf (w_37_0 * w_45_0) 1 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 0
    rw [s46, pairMul, rowOf_add, p12a, p24a, r45, t45, b45, a18, a31]
    rfl
  have k56 : w_56_0 = pw_31_0 := by
    rw [slice_rowOf (by inst_num) h56_0, slice_rowOf (by inst_num) ph31_0]
    show rowOf (w_37_0 * w_45_0) 2 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 1
    rw [s46', pairMul, rowOf_add, p12b, p24b, r45, t45, b45, a22, a35]
    rfl
  have k71 : w_71_0 = pw_46_0 := by
    rw [slice_rowOf (by inst_num) h71_0, slice_rowOf (by inst_num) ph46_0]
    show rowOf (w_37_0 * w_64_0) 1 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 0
    rw [s65, pairMul, rowOf_add, p12a', p24a', r64, t64, b64, a18, a31]
    rfl
  have k72 : w_72_0 = pw_47_0 := by
    rw [slice_rowOf (by inst_num) h72_0, slice_rowOf (by inst_num) ph47_0]
    show rowOf (w_37_0 * w_64_0) 2 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 1
    rw [s65', pairMul, rowOf_add, p12b', p24b', r64, t64, b64, a22, a35]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_76_0 = pw_33_0 := (hashSample_eq h76_0).trans (hashSample_eq ph33_0).symm
  have et1 : w_84_0 = pw_72_0 := (hashSample_eq h84_0).trans (hashSample_eq ph72_0).symm
  have eb0 : w_77_0 = pw_65_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_85_0 = pw_84_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h78_0
  rw [et1, eb1] at h86_0
  obtain ⟨t80, b80⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h78_0 h79_0 ph34_0 ph66_0
  have t80' := rebase_topRows h80_0 ph35_0 t80
  have b80' := rebase_bottomRows h80_0 ph67_0 b80
  obtain ⟨t88, b88⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h86_0 h87_0 ph73_0 ph85_0
  have t88' := rebase_topRows h88_0 ph74_0 t88
  have b88' := rebase_bottomRows h88_0 ph86_0 b88
  obtain ⟨-, s81, s81'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_80_0
  obtain ⟨-, s89, s89'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_88_0
  obtain ⟨q32a, q32b⟩ := stack2_mul_rows ph32_0 pw_35_0
  obtain ⟨q48a, q48b⟩ := stack2_mul_rows ph48_0 pw_67_0
  obtain ⟨q32a', q32b'⟩ := stack2_mul_rows ph32_0 pw_74_0
  obtain ⟨q48a', q48b'⟩ := stack2_mul_rows ph48_0 pw_86_0
  have k82 : w_82_0 = pw_70_0 := by
    rw [slice_rowOf (by inst_num) h82_0, slice_rowOf (by inst_num) ph70_0]
    show rowOf (w_74_0 * w_80_0) 1 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 0
    rw [s81, pairMul, rowOf_add, q32a, q48a, t80', b80', k55, k71]
    rfl
  have k92 : w_92_0 = pw_91_0 := by
    rw [slice_rowOf (by inst_num) h92_0, slice_rowOf (by inst_num) ph91_0]
    show rowOf (w_74_0 * w_80_0) 2 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 1
    rw [s81', pairMul, rowOf_add, q32b, q48b, t80', b80', k56, k72]
    rfl
  have k90 : w_90_0 = pw_89_0 := by
    rw [slice_rowOf (by inst_num) h90_0, slice_rowOf (by inst_num) ph89_0]
    show rowOf (w_74_0 * w_88_0) 1 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 0
    rw [s89, pairMul, rowOf_add, q32a', q48a', t88', b88', k55, k71]
    rfl
  have k94 : w_94_0 = pw_93_0 := by
    rw [slice_rowOf (by inst_num) h94_0, slice_rowOf (by inst_num) ph93_0]
    show rowOf (w_74_0 * w_88_0) 2 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 1
    rw [s89', pairMul, rowOf_add, q32b', q48b', t88', b88', k56, k72]
    rfl
  rw [k82] at h83_0
  rw [k92] at h93_0
  rw [k90] at h91_0
  rw [k94] at h95_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h83_0 ph71_0, blockModSwitchRuns_unique h91_0 ph90_0,
    blockModSwitchRuns_unique h93_0 ph92_0, blockModSwitchRuns_unique h95_0 ph94_0⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `0` of the fresh PRF (online `386`) are those the
preprocessing computes (`210`). -/
theorem keysCell386 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q7 N 1 14} {apay : Fin 5 → ExactMatrix Q7 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : ExactMatrix Q7 N 1 12} {bot0 bot1 : ExactMatrix Q6 N 1 10}
    {outs : Blocks Q6 10}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_388 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q7 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_212 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q7 N 1 14 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_388 at hon
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_6_0, hon⟩ := hon
  obtain ⟨w_7_0, hon⟩ := hon
  obtain ⟨w_7_concat_1, hon⟩ := hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  obtain ⟨w_14_concat_1, hon⟩ := hon
  obtain ⟨w_15_0, hon⟩ := hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_19_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hon
  obtain ⟨w_28_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  obtain ⟨w_30_0, hon⟩ := hon
  obtain ⟨w_30_concat_1, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  obtain ⟨w_34_0, hon⟩ := hon
  obtain ⟨w_35_0, hon⟩ := hon
  extract_lets -merge w_36_0 at hon
  obtain ⟨w_37_0, hon⟩ := hon
  extract_lets -merge w_40_0 at hon
  obtain ⟨w_41_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_43_concat_1, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_46_0, hon⟩ := hon
  obtain ⟨w_48_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_50_0, hon⟩ := hon
  extract_lets -merge w_51_0 at hon
  obtain ⟨w_52_0, hon⟩ := hon
  obtain ⟨w_53_0, hon⟩ := hon
  obtain ⟨w_54_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_58_0, hon⟩ := hon
  extract_lets -merge w_59_0 at hon
  obtain ⟨w_60_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  obtain ⟨w_65_0, hon⟩ := hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_67_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  obtain ⟨w_69_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_388.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_388.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_388.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_388.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_388.constraints_4] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, -, -, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, -, -, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_212 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_4_0 pw_10_0 pw_21_0 pw_24_0 pw_31_0 pw_38_0 pw_40_0 pw_58_0 pw_36_0 pw_39_0 pw_41_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_2_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_5_0, hpre⟩ := hpre
  extract_lets -merge pw_6_0 at hpre
  obtain ⟨pw_7_0, hpre⟩ := hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  extract_lets -merge pw_12_0 pw_13_0 at hpre
  obtain ⟨pw_14_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  obtain ⟨pw_17_0, hpre⟩ := hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  obtain ⟨pw_19_0, hpre⟩ := hpre
  extract_lets -merge pw_20_0 at hpre
  obtain ⟨pw_22_0, hpre⟩ := hpre
  extract_lets -merge pw_23_0 at hpre
  obtain ⟨pw_25_0, hpre⟩ := hpre
  extract_lets -merge pw_26_0 pw_27_0 at hpre
  obtain ⟨pw_28_0, hpre⟩ := hpre
  obtain ⟨pw_29_0, hpre⟩ := hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 at hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  extract_lets -merge pw_35_0 pw_37_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  extract_lets -merge pw_43_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  extract_lets -merge pw_47_0 at hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_49_0, hpre⟩ := hpre
  extract_lets -merge pw_50_0 pw_51_0 at hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  obtain ⟨pw_53_0, hpre⟩ := hpre
  obtain ⟨pw_54_0, hpre⟩ := hpre
  extract_lets -merge pw_63_0 at hpre
  obtain ⟨pw_55_0, hpre⟩ := hpre
  obtain ⟨pw_56_0, hpre⟩ := hpre
  extract_lets -merge pw_57_0 at hpre
  obtain ⟨pw_59_0, hpre⟩ := hpre
  extract_lets -merge pw_60_0 pw_61_0 pw_62_0 pw_64_0 at hpre
  obtain ⟨pw_65_0, hpre⟩ := hpre
  extract_lets -merge pw_66_0 at hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  obtain ⟨pw_68_0, hpre⟩ := hpre
  extract_lets -merge pw_69_0 pw_70_0 at hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  obtain ⟨pw_75_0, hpre⟩ := hpre
  obtain ⟨pw_76_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_212.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_212.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_212.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_212.constraints_3] at hpre
  obtain ⟨-, -, -, ph1_0, -, -, -, ph2_0, ph3_0, ph5_0, -, -, -, ph7_0, -, -, -, ph8_0, ph9_0, ph11_0, -, -, -, -, -, -, ph14_0, -, -, -, -, -, -, ph15_0, ph16_0, -, -, ph17_0, ph18_0, ph19_0, ph22_0, ph25_0, -, -, -, -, -, -, ph28_0, -, -, -, -, -, -, ph29_0, ph30_0, ph32_0, ph33_0, ph34_0, ph42_0, ph46_0, ph48_0, ph49_0, -, -, -, -, -, -, ph52_0, ph53_0, -, -, ph54_0, ph55_0, ph56_0, ph59_0, ph65_0, ph67_0, ph68_0, -, -, -, -, -, -, ph71_0, ph72_0, -, -, -, -, -, -, ph73_0, ph74_0, -, -, -, -, -, -, ph75_0, ph76_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h4_0 h6_0 h11_0 h13_0
  obtain ⟨hk, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_4_0 := by
    rw [hashSample_eq ph1_0, hw4, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e2 : pw_2_0 = w_6_0 := by
    rw [hashSample_eq ph2_0, hw6, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e7 : pw_7_0 = w_11_0 := by
    rw [hashSample_eq ph7_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e8 : pw_8_0 = w_13_0 := by
    rw [hashSample_eq ph8_0, hw13, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The public multiplication by the one encoding.
  obtain ⟨t20, b20⟩ := decompose_stack Backend.layout8 layoutQ7 rfl h18_0 h19_0 ph5_0 ph11_0
  have r20 := rebase_self (by inst_num) h20_0
  obtain ⟨t35, b35⟩ := decompose_stack Backend.layout8 layoutQ7 rfl h33_0 h34_0 ph22_0 ph25_0
  have r35 := rebase_self (by inst_num) h35_0
  obtain ⟨-, s21, s21'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_20_0
  obtain ⟨-, s36, s36'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_35_0
  obtain ⟨p3a, p3b⟩ := stack2_mul_rows ph3_0 pw_5_0
  obtain ⟨p9a, p9b⟩ := stack2_mul_rows ph9_0 pw_11_0
  obtain ⟨p3a', p3b'⟩ := stack2_mul_rows ph3_0 pw_22_0
  obtain ⟨p9a', p9b'⟩ := stack2_mul_rows ph9_0 pw_25_0
  have k28 : w_28_0 = pw_14_0 := by
    rw [slice_rowOf (by inst_num) h28_0, slice_rowOf (by inst_num) ph14_0]
    show rowOf (w_15_0 * w_20_0) 1 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 0
    rw [s21, pairMul, rowOf_add, p3a, p9a, r20, t20, b20, e1, e7]
    rfl
  have k29 : w_29_0 = pw_15_0 := by
    rw [slice_rowOf (by inst_num) h29_0, slice_rowOf (by inst_num) ph15_0]
    show rowOf (w_15_0 * w_20_0) 2 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 1
    rw [s21', pairMul, rowOf_add, p3b, p9b, r20, t20, b20, e2, e8]
    rfl
  have k41 : w_41_0 = pw_28_0 := by
    rw [slice_rowOf (by inst_num) h41_0, slice_rowOf (by inst_num) ph28_0]
    show rowOf (w_15_0 * w_35_0) 1 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 0
    rw [s36, pairMul, rowOf_add, p3a', p9a', r35, t35, b35, e1, e7]
    rfl
  have k42 : w_42_0 = pw_29_0 := by
    rw [slice_rowOf (by inst_num) h42_0, slice_rowOf (by inst_num) ph29_0]
    show rowOf (w_15_0 * w_35_0) 2 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 1
    rw [s36', pairMul, rowOf_add, p3b', p9b', r35, t35, b35, e2, e8]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_46_0 = pw_17_0 := (hashSample_eq h46_0).trans (hashSample_eq ph17_0).symm
  have et1 : w_54_0 = pw_54_0 := (hashSample_eq h54_0).trans (hashSample_eq ph54_0).symm
  have eb0 : w_47_0 = pw_47_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_55_0 = pw_66_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h48_0
  rw [et1, eb1] at h56_0
  obtain ⟨t50, b50⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h48_0 h49_0 ph18_0 ph48_0
  have t50' := rebase_topRows h50_0 ph19_0 t50
  have b50' := rebase_bottomRows h50_0 ph49_0 b50
  obtain ⟨t58, b58⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h56_0 h57_0 ph55_0 ph67_0
  have t58' := rebase_topRows h58_0 ph56_0 t58
  have b58' := rebase_bottomRows h58_0 ph68_0 b58
  obtain ⟨-, s51, s51'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_50_0
  obtain ⟨-, s59, s59'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_58_0
  obtain ⟨q16a, q16b⟩ := stack2_mul_rows ph16_0 pw_19_0
  obtain ⟨q30a, q30b⟩ := stack2_mul_rows ph30_0 pw_49_0
  obtain ⟨q16a', q16b'⟩ := stack2_mul_rows ph16_0 pw_56_0
  obtain ⟨q30a', q30b'⟩ := stack2_mul_rows ph30_0 pw_68_0
  have k52 : w_52_0 = pw_52_0 := by
    rw [slice_rowOf (by inst_num) h52_0, slice_rowOf (by inst_num) ph52_0]
    show rowOf (w_44_0 * w_50_0) 1 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 0
    rw [s51, pairMul, rowOf_add, q16a, q30a, t50', b50', k28, k41]
    rfl
  have k62 : w_62_0 = pw_73_0 := by
    rw [slice_rowOf (by inst_num) h62_0, slice_rowOf (by inst_num) ph73_0]
    show rowOf (w_44_0 * w_50_0) 2 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 1
    rw [s51', pairMul, rowOf_add, q16b, q30b, t50', b50', k29, k42]
    rfl
  have k60 : w_60_0 = pw_71_0 := by
    rw [slice_rowOf (by inst_num) h60_0, slice_rowOf (by inst_num) ph71_0]
    show rowOf (w_44_0 * w_58_0) 1 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 0
    rw [s59, pairMul, rowOf_add, q16a', q30a', t58', b58', k28, k41]
    rfl
  have k64 : w_64_0 = pw_75_0 := by
    rw [slice_rowOf (by inst_num) h64_0, slice_rowOf (by inst_num) ph75_0]
    show rowOf (w_44_0 * w_58_0) 2 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 1
    rw [s59', pairMul, rowOf_add, q16b', q30b', t58', b58', k29, k42]
    rfl
  rw [k52] at h53_0
  rw [k62] at h63_0
  rw [k60] at h61_0
  rw [k64] at h65_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h53_0 ph53_0, blockModSwitchRuns_unique h61_0 ph72_0,
    blockModSwitchRuns_unique h63_0 ph74_0, blockModSwitchRuns_unique h65_0 ph76_0⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `1` of the fresh PRF (online `390`) are those the
preprocessing computes (`214`). -/
theorem keysCell390 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q7 N 1 14} {apay : Fin 5 → ExactMatrix Q7 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : Fin 48 → ExactMatrix Q6 N 1 10}
    {bot0 bot1 : ExactMatrix Q5 N 1 8} {outs : Blocks Q5 8}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_392 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q7 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_216 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q7 N 1 14 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_392 at hon
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_5_0, hon⟩ := hon
  extract_lets -merge w_6_0 at hon
  obtain ⟨w_7_0, hon⟩ := hon
  extract_lets -merge w_8_0 at hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  extract_lets -merge w_15_0 at hon
  obtain ⟨w_16_0, hon⟩ := hon
  extract_lets -merge w_17_0 at hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_23_0, hon⟩ := hon
  obtain ⟨w_23_concat_1, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 at hon
  obtain ⟨w_27_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  extract_lets -merge w_30_0 at hon
  obtain ⟨w_31_0, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  extract_lets -merge w_34_0 at hon
  obtain ⟨w_35_0, hon⟩ := hon
  obtain ⟨w_36_0, hon⟩ := hon
  obtain ⟨w_36_concat_1, hon⟩ := hon
  obtain ⟨w_37_0, hon⟩ := hon
  obtain ⟨w_40_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_45_0, hon⟩ := hon
  extract_lets -merge w_46_0 at hon
  obtain ⟨w_47_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_51_0, hon⟩ := hon
  obtain ⟨w_52_0, hon⟩ := hon
  extract_lets -merge w_53_0 w_54_0 at hon
  obtain ⟨w_55_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_57_concat_1, hon⟩ := hon
  obtain ⟨w_59_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  extract_lets -merge w_65_0 at hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  extract_lets -merge w_69_0 w_70_0 at hon
  obtain ⟨w_71_0, hon⟩ := hon
  obtain ⟨w_72_0, hon⟩ := hon
  obtain ⟨w_73_0, hon⟩ := hon
  obtain ⟨w_73_concat_1, hon⟩ := hon
  obtain ⟨w_74_0, hon⟩ := hon
  obtain ⟨w_76_0, hon⟩ := hon
  obtain ⟨w_78_0, hon⟩ := hon
  obtain ⟨w_79_0, hon⟩ := hon
  obtain ⟨w_80_0, hon⟩ := hon
  extract_lets -merge w_81_0 at hon
  obtain ⟨w_82_0, hon⟩ := hon
  obtain ⟨w_83_0, hon⟩ := hon
  obtain ⟨w_84_0, hon⟩ := hon
  obtain ⟨w_86_0, hon⟩ := hon
  obtain ⟨w_87_0, hon⟩ := hon
  obtain ⟨w_88_0, hon⟩ := hon
  extract_lets -merge w_89_0 at hon
  obtain ⟨w_90_0, hon⟩ := hon
  obtain ⟨w_91_0, hon⟩ := hon
  obtain ⟨w_92_0, hon⟩ := hon
  obtain ⟨w_93_0, hon⟩ := hon
  obtain ⟨w_94_0, hon⟩ := hon
  obtain ⟨w_95_0, hon⟩ := hon
  obtain ⟨w_96_0, hon⟩ := hon
  obtain ⟨w_97_0, hon⟩ := hon
  obtain ⟨w_98_0, hon⟩ := hon
  obtain ⟨w_99_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_392.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_392.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_392.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_392.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_392.constraints_4, W_Stage_online_eval.parallel_parallel_generatedRoot_290_392.constraints_5] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, -, -, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, -, -, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_216 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_13_0 pw_25_0 pw_37_0 pw_41_0 pw_49_0 pw_56_0 pw_58_0 pw_76_0 pw_2_0 pw_14_0 pw_54_0 pw_57_0 pw_59_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_4_0, hpre⟩ := hpre
  extract_lets -merge pw_5_0 at hpre
  obtain ⟨pw_6_0, hpre⟩ := hpre
  extract_lets -merge pw_7_0 at hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  extract_lets -merge pw_10_0 at hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  obtain ⟨pw_12_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  extract_lets -merge pw_17_0 at hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  extract_lets -merge pw_19_0 at hpre
  obtain ⟨pw_20_0, hpre⟩ := hpre
  obtain ⟨pw_21_0, hpre⟩ := hpre
  extract_lets -merge pw_22_0 at hpre
  obtain ⟨pw_23_0, hpre⟩ := hpre
  obtain ⟨pw_24_0, hpre⟩ := hpre
  obtain ⟨pw_26_0, hpre⟩ := hpre
  obtain ⟨pw_27_0, hpre⟩ := hpre
  extract_lets -merge pw_28_0 pw_29_0 at hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_31_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  obtain ⟨pw_35_0, hpre⟩ := hpre
  extract_lets -merge pw_36_0 at hpre
  obtain ⟨pw_38_0, hpre⟩ := hpre
  obtain ⟨pw_39_0, hpre⟩ := hpre
  extract_lets -merge pw_40_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  obtain ⟨pw_43_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  obtain ⟨pw_47_0, hpre⟩ := hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_50_0, hpre⟩ := hpre
  extract_lets -merge pw_62_0 at hpre
  obtain ⟨pw_51_0, hpre⟩ := hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  extract_lets -merge pw_53_0 pw_55_0 at hpre
  obtain ⟨pw_60_0, hpre⟩ := hpre
  extract_lets -merge pw_61_0 pw_63_0 at hpre
  obtain ⟨pw_64_0, hpre⟩ := hpre
  extract_lets -merge pw_65_0 at hpre
  obtain ⟨pw_66_0, hpre⟩ := hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  extract_lets -merge pw_68_0 pw_69_0 at hpre
  obtain ⟨pw_70_0, hpre⟩ := hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  extract_lets -merge pw_81_0 at hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  extract_lets -merge pw_75_0 at hpre
  obtain ⟨pw_77_0, hpre⟩ := hpre
  extract_lets -merge pw_78_0 pw_79_0 pw_80_0 pw_82_0 at hpre
  obtain ⟨pw_83_0, hpre⟩ := hpre
  extract_lets -merge pw_84_0 at hpre
  obtain ⟨pw_85_0, hpre⟩ := hpre
  obtain ⟨pw_86_0, hpre⟩ := hpre
  extract_lets -merge pw_87_0 pw_88_0 at hpre
  obtain ⟨pw_89_0, hpre⟩ := hpre
  obtain ⟨pw_90_0, hpre⟩ := hpre
  obtain ⟨pw_91_0, hpre⟩ := hpre
  obtain ⟨pw_92_0, hpre⟩ := hpre
  obtain ⟨pw_93_0, hpre⟩ := hpre
  obtain ⟨pw_94_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_216.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_216.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_216.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_216.constraints_3] at hpre
  obtain ⟨-, -, -, ph1_0, ph3_0, ph4_0, ph6_0, ph8_0, -, -, -, ph9_0, ph11_0, ph12_0, -, ph14_0, ph14_0_1, ph15_0, ph16_0, -, -, -, ph18_0, ph20_0, -, -, -, ph21_0, ph23_0, ph24_0, ph14_0_2, ph14_0_3, ph26_0, ph27_0, -, -, -, -, -, -, ph30_0, -, -, -, -, -, -, ph31_0, ph32_0, -, -, ph33_0, ph34_0, ph35_0, ph14_0_4, ph14_0_5, ph38_0, ph39_0, ph14_0_6, ph14_0_7, ph42_0, ph43_0, -, -, -, -, -, -, ph46_0, -, -, -, -, -, -, ph47_0, ph48_0, ph50_0, ph51_0, ph52_0, ph60_0, ph64_0, ph66_0, ph67_0, -, -, -, -, -, -, ph70_0, ph71_0, -, -, ph72_0, ph73_0, ph74_0, ph77_0, ph83_0, ph85_0, ph86_0, -, -, -, -, -, -, ph89_0, ph90_0, -, -, -, -, -, -, ph91_0, ph92_0, -, -, -, -, -, -, ph93_0, ph94_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h11_0 h20_0 h29_0 h33_0
  obtain ⟨hk, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_11_0 := by
    rw [hashSample_eq ph1_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e9 : pw_9_0 = w_20_0 := by
    rw [hashSample_eq ph9_0, hw20, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e18 : pw_18_0 = w_29_0 := by
    rw [hashSample_eq ph18_0, hw29, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e21 : pw_21_0 = w_33_0 := by
    rw [hashSample_eq ph21_0, hw33, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The alignment `D(p G)` of the label.
  have g13 : w_13_0 = pw_3_0 := gadgetMatrixRuns_unique h13_0 ph3_0
  rw [g13] at h14_0
  have g14 : w_14_0 = pw_4_0 := centeredRebaseRuns_unique h14_0 ph4_0
  have g16 : w_16_0 = pw_6_0 := by
    apply gadgetDecomposeRuns_unique h16_0
    show gadgetDecomposeRuns _ _ _ (matrixMulScalarLeft _ w_14_0) _
    rw [g14]
    exact ph6_0
  have a18 : w_18_0 = pw_8_0 := by
    apply blockModSwitchRuns_unique h18_0
    show blockModSwitchRuns _ _ (w_11_0 * w_16_0) _
    rw [← e1, g16]
    exact ph8_0
  have a22 : w_22_0 = pw_11_0 := by
    apply blockModSwitchRuns_unique h22_0
    show blockModSwitchRuns _ _ (w_20_0 * w_16_0) _
    rw [← e9, g16]
    exact ph11_0
  have a31 : w_31_0 = pw_20_0 := by
    apply blockModSwitchRuns_unique h31_0
    show blockModSwitchRuns _ _ (w_29_0 * w_16_0) _
    rw [← e18, g16]
    exact ph20_0
  have a35 : w_35_0 = pw_23_0 := by
    apply blockModSwitchRuns_unique h35_0
    show blockModSwitchRuns _ _ (w_33_0 * w_16_0) _
    rw [← e21, g16]
    exact ph23_0
  -- The members of bucket `0`.
  have ho : w_39_0 = pw_14_0 := rfl
  rw [ho] at h40_0 h42_0 h59_0 h61_0
  have m40 : w_40_0 = pw_15_0 := familyGetDynamic_unique h40_0 ph15_0
  have m42 : w_42_0 = pw_26_0 := familyGetDynamic_unique h42_0 ph26_0
  have m59 : w_59_0 = pw_38_0 := familyGetDynamic_unique h59_0 ph38_0
  have m61 : w_61_0 = pw_42_0 := familyGetDynamic_unique h61_0 ph42_0
  rw [m40, m42] at h43_0
  rw [m59, m61] at h62_0
  obtain ⟨t45, b45⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h43_0 h44_0 ph16_0 ph27_0
  have r45 := rebase_self (by inst_num) h45_0
  obtain ⟨t64, b64⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h62_0 h63_0 ph39_0 ph43_0
  have r64 := rebase_self (by inst_num) h64_0
  obtain ⟨-, s46, s46'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_45_0
  obtain ⟨-, s65, s65'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_64_0
  obtain ⟨p12a, p12b⟩ := stack2_mul_rows ph12_0 pw_16_0
  obtain ⟨p24a, p24b⟩ := stack2_mul_rows ph24_0 pw_27_0
  obtain ⟨p12a', p12b'⟩ := stack2_mul_rows ph12_0 pw_39_0
  obtain ⟨p24a', p24b'⟩ := stack2_mul_rows ph24_0 pw_43_0
  have k55 : w_55_0 = pw_30_0 := by
    rw [slice_rowOf (by inst_num) h55_0, slice_rowOf (by inst_num) ph30_0]
    show rowOf (w_37_0 * w_45_0) 1 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 0
    rw [s46, pairMul, rowOf_add, p12a, p24a, r45, t45, b45, a18, a31]
    rfl
  have k56 : w_56_0 = pw_31_0 := by
    rw [slice_rowOf (by inst_num) h56_0, slice_rowOf (by inst_num) ph31_0]
    show rowOf (w_37_0 * w_45_0) 2 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 1
    rw [s46', pairMul, rowOf_add, p12b, p24b, r45, t45, b45, a22, a35]
    rfl
  have k71 : w_71_0 = pw_46_0 := by
    rw [slice_rowOf (by inst_num) h71_0, slice_rowOf (by inst_num) ph46_0]
    show rowOf (w_37_0 * w_64_0) 1 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 0
    rw [s65, pairMul, rowOf_add, p12a', p24a', r64, t64, b64, a18, a31]
    rfl
  have k72 : w_72_0 = pw_47_0 := by
    rw [slice_rowOf (by inst_num) h72_0, slice_rowOf (by inst_num) ph47_0]
    show rowOf (w_37_0 * w_64_0) 2 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 1
    rw [s65', pairMul, rowOf_add, p12b', p24b', r64, t64, b64, a22, a35]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_76_0 = pw_33_0 := (hashSample_eq h76_0).trans (hashSample_eq ph33_0).symm
  have et1 : w_84_0 = pw_72_0 := (hashSample_eq h84_0).trans (hashSample_eq ph72_0).symm
  have eb0 : w_77_0 = pw_65_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_85_0 = pw_84_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h78_0
  rw [et1, eb1] at h86_0
  obtain ⟨t80, b80⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h78_0 h79_0 ph34_0 ph66_0
  have t80' := rebase_topRows h80_0 ph35_0 t80
  have b80' := rebase_bottomRows h80_0 ph67_0 b80
  obtain ⟨t88, b88⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h86_0 h87_0 ph73_0 ph85_0
  have t88' := rebase_topRows h88_0 ph74_0 t88
  have b88' := rebase_bottomRows h88_0 ph86_0 b88
  obtain ⟨-, s81, s81'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_80_0
  obtain ⟨-, s89, s89'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_88_0
  obtain ⟨q32a, q32b⟩ := stack2_mul_rows ph32_0 pw_35_0
  obtain ⟨q48a, q48b⟩ := stack2_mul_rows ph48_0 pw_67_0
  obtain ⟨q32a', q32b'⟩ := stack2_mul_rows ph32_0 pw_74_0
  obtain ⟨q48a', q48b'⟩ := stack2_mul_rows ph48_0 pw_86_0
  have k82 : w_82_0 = pw_70_0 := by
    rw [slice_rowOf (by inst_num) h82_0, slice_rowOf (by inst_num) ph70_0]
    show rowOf (w_74_0 * w_80_0) 1 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 0
    rw [s81, pairMul, rowOf_add, q32a, q48a, t80', b80', k55, k71]
    rfl
  have k92 : w_92_0 = pw_91_0 := by
    rw [slice_rowOf (by inst_num) h92_0, slice_rowOf (by inst_num) ph91_0]
    show rowOf (w_74_0 * w_80_0) 2 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 1
    rw [s81', pairMul, rowOf_add, q32b, q48b, t80', b80', k56, k72]
    rfl
  have k90 : w_90_0 = pw_89_0 := by
    rw [slice_rowOf (by inst_num) h90_0, slice_rowOf (by inst_num) ph89_0]
    show rowOf (w_74_0 * w_88_0) 1 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 0
    rw [s89, pairMul, rowOf_add, q32a', q48a', t88', b88', k55, k71]
    rfl
  have k94 : w_94_0 = pw_93_0 := by
    rw [slice_rowOf (by inst_num) h94_0, slice_rowOf (by inst_num) ph93_0]
    show rowOf (w_74_0 * w_88_0) 2 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 1
    rw [s89', pairMul, rowOf_add, q32b', q48b', t88', b88', k56, k72]
    rfl
  rw [k82] at h83_0
  rw [k92] at h93_0
  rw [k90] at h91_0
  rw [k94] at h95_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h83_0 ph71_0, blockModSwitchRuns_unique h91_0 ph90_0,
    blockModSwitchRuns_unique h93_0 ph92_0, blockModSwitchRuns_unique h95_0 ph94_0⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `0` of the smudge PRF (online `554`) are those the
preprocessing computes (`20`). -/
theorem keysCell554 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12} {apay : Fin 5 → ExactMatrix Q6 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : ExactMatrix Q6 N 1 10} {bot0 bot1 : ExactMatrix Q5 N 1 8}
    {outs : Blocks Q5 8}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_556 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q6 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_20 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q6 N 1 12 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_556 at hon
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_6_0, hon⟩ := hon
  obtain ⟨w_7_0, hon⟩ := hon
  obtain ⟨w_7_concat_1, hon⟩ := hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  obtain ⟨w_14_concat_1, hon⟩ := hon
  obtain ⟨w_15_0, hon⟩ := hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_19_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hon
  obtain ⟨w_28_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  obtain ⟨w_30_0, hon⟩ := hon
  obtain ⟨w_30_concat_1, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  obtain ⟨w_34_0, hon⟩ := hon
  obtain ⟨w_35_0, hon⟩ := hon
  extract_lets -merge w_36_0 at hon
  obtain ⟨w_37_0, hon⟩ := hon
  extract_lets -merge w_40_0 at hon
  obtain ⟨w_41_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_43_concat_1, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_46_0, hon⟩ := hon
  obtain ⟨w_48_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_50_0, hon⟩ := hon
  extract_lets -merge w_51_0 at hon
  obtain ⟨w_52_0, hon⟩ := hon
  obtain ⟨w_53_0, hon⟩ := hon
  obtain ⟨w_54_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_58_0, hon⟩ := hon
  extract_lets -merge w_59_0 at hon
  obtain ⟨w_60_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  obtain ⟨w_65_0, hon⟩ := hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_67_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  obtain ⟨w_69_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_556.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_556.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_556.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_556.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_556.constraints_4] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, -, -, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, -, -, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_20 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_4_0 pw_10_0 pw_21_0 pw_24_0 pw_31_0 pw_38_0 pw_40_0 pw_58_0 pw_36_0 pw_39_0 pw_41_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_2_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_5_0, hpre⟩ := hpre
  extract_lets -merge pw_6_0 at hpre
  obtain ⟨pw_7_0, hpre⟩ := hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  extract_lets -merge pw_12_0 pw_13_0 at hpre
  obtain ⟨pw_14_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  obtain ⟨pw_17_0, hpre⟩ := hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  obtain ⟨pw_19_0, hpre⟩ := hpre
  extract_lets -merge pw_20_0 at hpre
  obtain ⟨pw_22_0, hpre⟩ := hpre
  extract_lets -merge pw_23_0 at hpre
  obtain ⟨pw_25_0, hpre⟩ := hpre
  extract_lets -merge pw_26_0 pw_27_0 at hpre
  obtain ⟨pw_28_0, hpre⟩ := hpre
  obtain ⟨pw_29_0, hpre⟩ := hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 at hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  extract_lets -merge pw_35_0 pw_37_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  extract_lets -merge pw_43_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  extract_lets -merge pw_47_0 at hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_49_0, hpre⟩ := hpre
  extract_lets -merge pw_50_0 pw_51_0 at hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  obtain ⟨pw_53_0, hpre⟩ := hpre
  obtain ⟨pw_54_0, hpre⟩ := hpre
  extract_lets -merge pw_63_0 at hpre
  obtain ⟨pw_55_0, hpre⟩ := hpre
  obtain ⟨pw_56_0, hpre⟩ := hpre
  extract_lets -merge pw_57_0 at hpre
  obtain ⟨pw_59_0, hpre⟩ := hpre
  extract_lets -merge pw_60_0 pw_61_0 pw_62_0 pw_64_0 at hpre
  obtain ⟨pw_65_0, hpre⟩ := hpre
  extract_lets -merge pw_66_0 at hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  obtain ⟨pw_68_0, hpre⟩ := hpre
  extract_lets -merge pw_69_0 pw_70_0 at hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  obtain ⟨pw_75_0, hpre⟩ := hpre
  obtain ⟨pw_76_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_20.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_20.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_20.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_20.constraints_3] at hpre
  obtain ⟨-, -, -, ph1_0, -, -, -, ph2_0, ph3_0, ph5_0, -, -, -, ph7_0, -, -, -, ph8_0, ph9_0, ph11_0, -, -, -, -, -, -, ph14_0, -, -, -, -, -, -, ph15_0, ph16_0, -, -, ph17_0, ph18_0, ph19_0, ph22_0, ph25_0, -, -, -, -, -, -, ph28_0, -, -, -, -, -, -, ph29_0, ph30_0, ph32_0, ph33_0, ph34_0, ph42_0, ph46_0, ph48_0, ph49_0, -, -, -, -, -, -, ph52_0, ph53_0, -, -, ph54_0, ph55_0, ph56_0, ph59_0, ph65_0, ph67_0, ph68_0, -, -, -, -, -, -, ph71_0, ph72_0, -, -, -, -, -, -, ph73_0, ph74_0, -, -, -, -, -, -, ph75_0, ph76_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h4_0 h6_0 h11_0 h13_0
  obtain ⟨hk, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_4_0 := by
    rw [hashSample_eq ph1_0, hw4, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e2 : pw_2_0 = w_6_0 := by
    rw [hashSample_eq ph2_0, hw6, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e7 : pw_7_0 = w_11_0 := by
    rw [hashSample_eq ph7_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e8 : pw_8_0 = w_13_0 := by
    rw [hashSample_eq ph8_0, hw13, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The public multiplication by the one encoding.
  obtain ⟨t20, b20⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h18_0 h19_0 ph5_0 ph11_0
  have r20 := rebase_self (by inst_num) h20_0
  obtain ⟨t35, b35⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h33_0 h34_0 ph22_0 ph25_0
  have r35 := rebase_self (by inst_num) h35_0
  obtain ⟨-, s21, s21'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_20_0
  obtain ⟨-, s36, s36'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_35_0
  obtain ⟨p3a, p3b⟩ := stack2_mul_rows ph3_0 pw_5_0
  obtain ⟨p9a, p9b⟩ := stack2_mul_rows ph9_0 pw_11_0
  obtain ⟨p3a', p3b'⟩ := stack2_mul_rows ph3_0 pw_22_0
  obtain ⟨p9a', p9b'⟩ := stack2_mul_rows ph9_0 pw_25_0
  have k28 : w_28_0 = pw_14_0 := by
    rw [slice_rowOf (by inst_num) h28_0, slice_rowOf (by inst_num) ph14_0]
    show rowOf (w_15_0 * w_20_0) 1 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 0
    rw [s21, pairMul, rowOf_add, p3a, p9a, r20, t20, b20, e1, e7]
    rfl
  have k29 : w_29_0 = pw_15_0 := by
    rw [slice_rowOf (by inst_num) h29_0, slice_rowOf (by inst_num) ph15_0]
    show rowOf (w_15_0 * w_20_0) 2 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 1
    rw [s21', pairMul, rowOf_add, p3b, p9b, r20, t20, b20, e2, e8]
    rfl
  have k41 : w_41_0 = pw_28_0 := by
    rw [slice_rowOf (by inst_num) h41_0, slice_rowOf (by inst_num) ph28_0]
    show rowOf (w_15_0 * w_35_0) 1 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 0
    rw [s36, pairMul, rowOf_add, p3a', p9a', r35, t35, b35, e1, e7]
    rfl
  have k42 : w_42_0 = pw_29_0 := by
    rw [slice_rowOf (by inst_num) h42_0, slice_rowOf (by inst_num) ph29_0]
    show rowOf (w_15_0 * w_35_0) 2 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 1
    rw [s36', pairMul, rowOf_add, p3b', p9b', r35, t35, b35, e2, e8]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_46_0 = pw_17_0 := (hashSample_eq h46_0).trans (hashSample_eq ph17_0).symm
  have et1 : w_54_0 = pw_54_0 := (hashSample_eq h54_0).trans (hashSample_eq ph54_0).symm
  have eb0 : w_47_0 = pw_47_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_55_0 = pw_66_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h48_0
  rw [et1, eb1] at h56_0
  obtain ⟨t50, b50⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h48_0 h49_0 ph18_0 ph48_0
  have t50' := rebase_topRows h50_0 ph19_0 t50
  have b50' := rebase_bottomRows h50_0 ph49_0 b50
  obtain ⟨t58, b58⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h56_0 h57_0 ph55_0 ph67_0
  have t58' := rebase_topRows h58_0 ph56_0 t58
  have b58' := rebase_bottomRows h58_0 ph68_0 b58
  obtain ⟨-, s51, s51'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_50_0
  obtain ⟨-, s59, s59'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_58_0
  obtain ⟨q16a, q16b⟩ := stack2_mul_rows ph16_0 pw_19_0
  obtain ⟨q30a, q30b⟩ := stack2_mul_rows ph30_0 pw_49_0
  obtain ⟨q16a', q16b'⟩ := stack2_mul_rows ph16_0 pw_56_0
  obtain ⟨q30a', q30b'⟩ := stack2_mul_rows ph30_0 pw_68_0
  have k52 : w_52_0 = pw_52_0 := by
    rw [slice_rowOf (by inst_num) h52_0, slice_rowOf (by inst_num) ph52_0]
    show rowOf (w_44_0 * w_50_0) 1 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 0
    rw [s51, pairMul, rowOf_add, q16a, q30a, t50', b50', k28, k41]
    rfl
  have k62 : w_62_0 = pw_73_0 := by
    rw [slice_rowOf (by inst_num) h62_0, slice_rowOf (by inst_num) ph73_0]
    show rowOf (w_44_0 * w_50_0) 2 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 1
    rw [s51', pairMul, rowOf_add, q16b, q30b, t50', b50', k29, k42]
    rfl
  have k60 : w_60_0 = pw_71_0 := by
    rw [slice_rowOf (by inst_num) h60_0, slice_rowOf (by inst_num) ph71_0]
    show rowOf (w_44_0 * w_58_0) 1 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 0
    rw [s59, pairMul, rowOf_add, q16a', q30a', t58', b58', k28, k41]
    rfl
  have k64 : w_64_0 = pw_75_0 := by
    rw [slice_rowOf (by inst_num) h64_0, slice_rowOf (by inst_num) ph75_0]
    show rowOf (w_44_0 * w_58_0) 2 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 1
    rw [s59', pairMul, rowOf_add, q16b', q30b', t58', b58', k29, k42]
    rfl
  rw [k52] at h53_0
  rw [k62] at h63_0
  rw [k60] at h61_0
  rw [k64] at h65_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h53_0 ph53_0, blockModSwitchRuns_unique h61_0 ph72_0,
    blockModSwitchRuns_unique h63_0 ph74_0, blockModSwitchRuns_unique h65_0 ph76_0⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `1` of the smudge PRF (online `558`) are those the
preprocessing computes (`24`). -/
theorem keysCell558 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12} {apay : Fin 5 → ExactMatrix Q6 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : Fin 160 → ExactMatrix Q5 N 1 8}
    {bot0 bot1 : ExactMatrix Q0 N 1 6} {outs : Blocks Q0 6}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_560 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q6 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_24 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q6 N 1 12 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_560 at hon
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_5_0, hon⟩ := hon
  extract_lets -merge w_6_0 at hon
  obtain ⟨w_7_0, hon⟩ := hon
  extract_lets -merge w_8_0 at hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  extract_lets -merge w_15_0 at hon
  obtain ⟨w_16_0, hon⟩ := hon
  extract_lets -merge w_17_0 at hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_23_0, hon⟩ := hon
  obtain ⟨w_23_concat_1, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 at hon
  obtain ⟨w_27_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  extract_lets -merge w_30_0 at hon
  obtain ⟨w_31_0, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  extract_lets -merge w_34_0 at hon
  obtain ⟨w_35_0, hon⟩ := hon
  obtain ⟨w_36_0, hon⟩ := hon
  obtain ⟨w_36_concat_1, hon⟩ := hon
  obtain ⟨w_37_0, hon⟩ := hon
  obtain ⟨w_40_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_45_0, hon⟩ := hon
  extract_lets -merge w_46_0 at hon
  obtain ⟨w_47_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_51_0, hon⟩ := hon
  obtain ⟨w_52_0, hon⟩ := hon
  extract_lets -merge w_53_0 w_54_0 at hon
  obtain ⟨w_55_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_57_concat_1, hon⟩ := hon
  obtain ⟨w_59_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  extract_lets -merge w_65_0 at hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  extract_lets -merge w_69_0 w_70_0 at hon
  obtain ⟨w_71_0, hon⟩ := hon
  obtain ⟨w_72_0, hon⟩ := hon
  obtain ⟨w_73_0, hon⟩ := hon
  obtain ⟨w_73_concat_1, hon⟩ := hon
  obtain ⟨w_74_0, hon⟩ := hon
  obtain ⟨w_76_0, hon⟩ := hon
  obtain ⟨w_78_0, hon⟩ := hon
  obtain ⟨w_79_0, hon⟩ := hon
  obtain ⟨w_80_0, hon⟩ := hon
  extract_lets -merge w_81_0 at hon
  obtain ⟨w_82_0, hon⟩ := hon
  obtain ⟨w_83_0, hon⟩ := hon
  obtain ⟨w_84_0, hon⟩ := hon
  obtain ⟨w_86_0, hon⟩ := hon
  obtain ⟨w_87_0, hon⟩ := hon
  obtain ⟨w_88_0, hon⟩ := hon
  extract_lets -merge w_89_0 at hon
  obtain ⟨w_90_0, hon⟩ := hon
  obtain ⟨w_91_0, hon⟩ := hon
  obtain ⟨w_92_0, hon⟩ := hon
  obtain ⟨w_93_0, hon⟩ := hon
  obtain ⟨w_94_0, hon⟩ := hon
  obtain ⟨w_95_0, hon⟩ := hon
  obtain ⟨w_96_0, hon⟩ := hon
  obtain ⟨w_97_0, hon⟩ := hon
  obtain ⟨w_98_0, hon⟩ := hon
  obtain ⟨w_99_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_560.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_560.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_560.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_560.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_560.constraints_4, W_Stage_online_eval.parallel_parallel_generatedRoot_290_560.constraints_5] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, -, -, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, -, -, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_24 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_13_0 pw_25_0 pw_37_0 pw_41_0 pw_49_0 pw_56_0 pw_58_0 pw_76_0 pw_2_0 pw_14_0 pw_54_0 pw_57_0 pw_59_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_4_0, hpre⟩ := hpre
  extract_lets -merge pw_5_0 at hpre
  obtain ⟨pw_6_0, hpre⟩ := hpre
  extract_lets -merge pw_7_0 at hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  extract_lets -merge pw_10_0 at hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  obtain ⟨pw_12_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  extract_lets -merge pw_17_0 at hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  extract_lets -merge pw_19_0 at hpre
  obtain ⟨pw_20_0, hpre⟩ := hpre
  obtain ⟨pw_21_0, hpre⟩ := hpre
  extract_lets -merge pw_22_0 at hpre
  obtain ⟨pw_23_0, hpre⟩ := hpre
  obtain ⟨pw_24_0, hpre⟩ := hpre
  obtain ⟨pw_26_0, hpre⟩ := hpre
  obtain ⟨pw_27_0, hpre⟩ := hpre
  extract_lets -merge pw_28_0 pw_29_0 at hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_31_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  obtain ⟨pw_35_0, hpre⟩ := hpre
  extract_lets -merge pw_36_0 at hpre
  obtain ⟨pw_38_0, hpre⟩ := hpre
  obtain ⟨pw_39_0, hpre⟩ := hpre
  extract_lets -merge pw_40_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  obtain ⟨pw_43_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  obtain ⟨pw_47_0, hpre⟩ := hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_50_0, hpre⟩ := hpre
  extract_lets -merge pw_62_0 at hpre
  obtain ⟨pw_51_0, hpre⟩ := hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  extract_lets -merge pw_53_0 pw_55_0 at hpre
  obtain ⟨pw_60_0, hpre⟩ := hpre
  extract_lets -merge pw_61_0 pw_63_0 at hpre
  obtain ⟨pw_64_0, hpre⟩ := hpre
  extract_lets -merge pw_65_0 at hpre
  obtain ⟨pw_66_0, hpre⟩ := hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  extract_lets -merge pw_68_0 pw_69_0 at hpre
  obtain ⟨pw_70_0, hpre⟩ := hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  extract_lets -merge pw_81_0 at hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  extract_lets -merge pw_75_0 at hpre
  obtain ⟨pw_77_0, hpre⟩ := hpre
  extract_lets -merge pw_78_0 pw_79_0 pw_80_0 pw_82_0 at hpre
  obtain ⟨pw_83_0, hpre⟩ := hpre
  extract_lets -merge pw_84_0 at hpre
  obtain ⟨pw_85_0, hpre⟩ := hpre
  obtain ⟨pw_86_0, hpre⟩ := hpre
  extract_lets -merge pw_87_0 pw_88_0 at hpre
  obtain ⟨pw_89_0, hpre⟩ := hpre
  obtain ⟨pw_90_0, hpre⟩ := hpre
  obtain ⟨pw_91_0, hpre⟩ := hpre
  obtain ⟨pw_92_0, hpre⟩ := hpre
  obtain ⟨pw_93_0, hpre⟩ := hpre
  obtain ⟨pw_94_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_24.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_24.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_24.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_24.constraints_3] at hpre
  obtain ⟨-, -, -, ph1_0, ph3_0, ph4_0, ph6_0, ph8_0, -, -, -, ph9_0, ph11_0, ph12_0, -, ph14_0, ph14_0_1, ph15_0, ph16_0, -, -, -, ph18_0, ph20_0, -, -, -, ph21_0, ph23_0, ph24_0, ph14_0_2, ph14_0_3, ph26_0, ph27_0, -, -, -, -, -, -, ph30_0, -, -, -, -, -, -, ph31_0, ph32_0, -, -, ph33_0, ph34_0, ph35_0, ph14_0_4, ph14_0_5, ph38_0, ph39_0, ph14_0_6, ph14_0_7, ph42_0, ph43_0, -, -, -, -, -, -, ph46_0, -, -, -, -, -, -, ph47_0, ph48_0, ph50_0, ph51_0, ph52_0, -, -, ph60_0, ph64_0, ph66_0, ph67_0, -, -, -, -, -, -, ph70_0, ph71_0, -, -, ph72_0, ph73_0, ph74_0, ph77_0, ph83_0, ph85_0, ph86_0, -, -, -, -, -, -, ph89_0, ph90_0, -, -, -, -, -, -, ph91_0, ph92_0, -, -, -, -, -, -, ph93_0, ph94_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h11_0 h20_0 h29_0 h33_0
  obtain ⟨hk, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_11_0 := by
    rw [hashSample_eq ph1_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e9 : pw_9_0 = w_20_0 := by
    rw [hashSample_eq ph9_0, hw20, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e18 : pw_18_0 = w_29_0 := by
    rw [hashSample_eq ph18_0, hw29, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e21 : pw_21_0 = w_33_0 := by
    rw [hashSample_eq ph21_0, hw33, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The alignment `D(p G)` of the label.
  have g13 : w_13_0 = pw_3_0 := gadgetMatrixRuns_unique h13_0 ph3_0
  rw [g13] at h14_0
  have g14 : w_14_0 = pw_4_0 := centeredRebaseRuns_unique h14_0 ph4_0
  have g16 : w_16_0 = pw_6_0 := by
    apply gadgetDecomposeRuns_unique h16_0
    show gadgetDecomposeRuns _ _ _ (matrixMulScalarLeft _ w_14_0) _
    rw [g14]
    exact ph6_0
  have a18 : w_18_0 = pw_8_0 := by
    apply blockModSwitchRuns_unique h18_0
    show blockModSwitchRuns _ _ (w_11_0 * w_16_0) _
    rw [← e1, g16]
    exact ph8_0
  have a22 : w_22_0 = pw_11_0 := by
    apply blockModSwitchRuns_unique h22_0
    show blockModSwitchRuns _ _ (w_20_0 * w_16_0) _
    rw [← e9, g16]
    exact ph11_0
  have a31 : w_31_0 = pw_20_0 := by
    apply blockModSwitchRuns_unique h31_0
    show blockModSwitchRuns _ _ (w_29_0 * w_16_0) _
    rw [← e18, g16]
    exact ph20_0
  have a35 : w_35_0 = pw_23_0 := by
    apply blockModSwitchRuns_unique h35_0
    show blockModSwitchRuns _ _ (w_33_0 * w_16_0) _
    rw [← e21, g16]
    exact ph23_0
  -- The members of bucket `0`.
  have ho : w_39_0 = pw_14_0 := rfl
  rw [ho] at h40_0 h42_0 h59_0 h61_0
  have m40 : w_40_0 = pw_15_0 := familyGetDynamic_unique h40_0 ph15_0
  have m42 : w_42_0 = pw_26_0 := familyGetDynamic_unique h42_0 ph26_0
  have m59 : w_59_0 = pw_38_0 := familyGetDynamic_unique h59_0 ph38_0
  have m61 : w_61_0 = pw_42_0 := familyGetDynamic_unique h61_0 ph42_0
  rw [m40, m42] at h43_0
  rw [m59, m61] at h62_0
  obtain ⟨t45, b45⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h43_0 h44_0 ph16_0 ph27_0
  have r45 := rebase_self (by inst_num) h45_0
  obtain ⟨t64, b64⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h62_0 h63_0 ph39_0 ph43_0
  have r64 := rebase_self (by inst_num) h64_0
  obtain ⟨-, s46, s46'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_45_0
  obtain ⟨-, s65, s65'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_64_0
  obtain ⟨p12a, p12b⟩ := stack2_mul_rows ph12_0 pw_16_0
  obtain ⟨p24a, p24b⟩ := stack2_mul_rows ph24_0 pw_27_0
  obtain ⟨p12a', p12b'⟩ := stack2_mul_rows ph12_0 pw_39_0
  obtain ⟨p24a', p24b'⟩ := stack2_mul_rows ph24_0 pw_43_0
  have k55 : w_55_0 = pw_30_0 := by
    rw [slice_rowOf (by inst_num) h55_0, slice_rowOf (by inst_num) ph30_0]
    show rowOf (w_37_0 * w_45_0) 1 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 0
    rw [s46, pairMul, rowOf_add, p12a, p24a, r45, t45, b45, a18, a31]
    rfl
  have k56 : w_56_0 = pw_31_0 := by
    rw [slice_rowOf (by inst_num) h56_0, slice_rowOf (by inst_num) ph31_0]
    show rowOf (w_37_0 * w_45_0) 2 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 1
    rw [s46', pairMul, rowOf_add, p12b, p24b, r45, t45, b45, a22, a35]
    rfl
  have k71 : w_71_0 = pw_46_0 := by
    rw [slice_rowOf (by inst_num) h71_0, slice_rowOf (by inst_num) ph46_0]
    show rowOf (w_37_0 * w_64_0) 1 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 0
    rw [s65, pairMul, rowOf_add, p12a', p24a', r64, t64, b64, a18, a31]
    rfl
  have k72 : w_72_0 = pw_47_0 := by
    rw [slice_rowOf (by inst_num) h72_0, slice_rowOf (by inst_num) ph47_0]
    show rowOf (w_37_0 * w_64_0) 2 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 1
    rw [s65', pairMul, rowOf_add, p12b', p24b', r64, t64, b64, a22, a35]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_76_0 = pw_33_0 := (hashSample_eq h76_0).trans (hashSample_eq ph33_0).symm
  have et1 : w_84_0 = pw_72_0 := (hashSample_eq h84_0).trans (hashSample_eq ph72_0).symm
  have eb0 : w_77_0 = pw_65_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_85_0 = pw_84_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h78_0
  rw [et1, eb1] at h86_0
  obtain ⟨t80, b80⟩ := decompose_stack Backend.layout5 layoutQ0 rfl h78_0 h79_0 ph34_0 ph66_0
  have t80' := rebase_topRows h80_0 ph35_0 t80
  have b80' := rebase_bottomRows h80_0 ph67_0 b80
  obtain ⟨t88, b88⟩ := decompose_stack Backend.layout5 layoutQ0 rfl h86_0 h87_0 ph73_0 ph85_0
  have t88' := rebase_topRows h88_0 ph74_0 t88
  have b88' := rebase_bottomRows h88_0 ph86_0 b88
  obtain ⟨-, s81, s81'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_80_0
  obtain ⟨-, s89, s89'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_88_0
  obtain ⟨q32a, q32b⟩ := stack2_mul_rows ph32_0 pw_35_0
  obtain ⟨q48a, q48b⟩ := stack2_mul_rows ph48_0 pw_67_0
  obtain ⟨q32a', q32b'⟩ := stack2_mul_rows ph32_0 pw_74_0
  obtain ⟨q48a', q48b'⟩ := stack2_mul_rows ph48_0 pw_86_0
  have k82 : w_82_0 = pw_70_0 := by
    rw [slice_rowOf (by inst_num) h82_0, slice_rowOf (by inst_num) ph70_0]
    show rowOf (w_74_0 * w_80_0) 1 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 0
    rw [s81, pairMul, rowOf_add, q32a, q48a, t80', b80', k55, k71]
    rfl
  have k92 : w_92_0 = pw_91_0 := by
    rw [slice_rowOf (by inst_num) h92_0, slice_rowOf (by inst_num) ph91_0]
    show rowOf (w_74_0 * w_80_0) 2 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 1
    rw [s81', pairMul, rowOf_add, q32b, q48b, t80', b80', k56, k72]
    rfl
  have k90 : w_90_0 = pw_89_0 := by
    rw [slice_rowOf (by inst_num) h90_0, slice_rowOf (by inst_num) ph89_0]
    show rowOf (w_74_0 * w_88_0) 1 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 0
    rw [s89, pairMul, rowOf_add, q32a', q48a', t88', b88', k55, k71]
    rfl
  have k94 : w_94_0 = pw_93_0 := by
    rw [slice_rowOf (by inst_num) h94_0, slice_rowOf (by inst_num) ph93_0]
    show rowOf (w_74_0 * w_88_0) 2 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 1
    rw [s89', pairMul, rowOf_add, q32b', q48b', t88', b88', k56, k72]
    rfl
  rw [k82] at h83_0
  rw [k92] at h93_0
  rw [k90] at h91_0
  rw [k94] at h95_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h83_0 ph71_0, blockModSwitchRuns_unique h91_0 ph90_0,
    blockModSwitchRuns_unique h93_0 ph92_0, blockModSwitchRuns_unique h95_0 ph94_0⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `0` of the smudge PRF (online `609`) are those the
preprocessing computes (`381`). -/
theorem keysCell609 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12} {apay : Fin 5 → ExactMatrix Q6 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : ExactMatrix Q6 N 1 10} {bot0 bot1 : ExactMatrix Q5 N 1 8}
    {outs : Blocks Q5 8}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_611 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q6 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_385 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q6 N 1 12 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_611 at hon
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_6_0, hon⟩ := hon
  obtain ⟨w_7_0, hon⟩ := hon
  obtain ⟨w_7_concat_1, hon⟩ := hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  obtain ⟨w_14_concat_1, hon⟩ := hon
  obtain ⟨w_15_0, hon⟩ := hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_19_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hon
  obtain ⟨w_28_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  obtain ⟨w_30_0, hon⟩ := hon
  obtain ⟨w_30_concat_1, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  obtain ⟨w_34_0, hon⟩ := hon
  obtain ⟨w_35_0, hon⟩ := hon
  extract_lets -merge w_36_0 at hon
  obtain ⟨w_37_0, hon⟩ := hon
  extract_lets -merge w_40_0 at hon
  obtain ⟨w_41_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_43_concat_1, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_46_0, hon⟩ := hon
  obtain ⟨w_48_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_50_0, hon⟩ := hon
  extract_lets -merge w_51_0 at hon
  obtain ⟨w_52_0, hon⟩ := hon
  obtain ⟨w_53_0, hon⟩ := hon
  obtain ⟨w_54_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_58_0, hon⟩ := hon
  extract_lets -merge w_59_0 at hon
  obtain ⟨w_60_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  obtain ⟨w_65_0, hon⟩ := hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_67_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  obtain ⟨w_69_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_611.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_611.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_611.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_611.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_611.constraints_4] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, -, -, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, -, -, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_385 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_4_0 pw_10_0 pw_21_0 pw_24_0 pw_31_0 pw_38_0 pw_40_0 pw_58_0 pw_36_0 pw_39_0 pw_41_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_2_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_5_0, hpre⟩ := hpre
  extract_lets -merge pw_6_0 at hpre
  obtain ⟨pw_7_0, hpre⟩ := hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  extract_lets -merge pw_12_0 pw_13_0 at hpre
  obtain ⟨pw_14_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  obtain ⟨pw_17_0, hpre⟩ := hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  obtain ⟨pw_19_0, hpre⟩ := hpre
  extract_lets -merge pw_20_0 at hpre
  obtain ⟨pw_22_0, hpre⟩ := hpre
  extract_lets -merge pw_23_0 at hpre
  obtain ⟨pw_25_0, hpre⟩ := hpre
  extract_lets -merge pw_26_0 pw_27_0 at hpre
  obtain ⟨pw_28_0, hpre⟩ := hpre
  obtain ⟨pw_29_0, hpre⟩ := hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 at hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  extract_lets -merge pw_35_0 pw_37_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  extract_lets -merge pw_43_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  extract_lets -merge pw_47_0 at hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_49_0, hpre⟩ := hpre
  extract_lets -merge pw_50_0 pw_51_0 at hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  obtain ⟨pw_53_0, hpre⟩ := hpre
  obtain ⟨pw_54_0, hpre⟩ := hpre
  extract_lets -merge pw_63_0 at hpre
  obtain ⟨pw_55_0, hpre⟩ := hpre
  obtain ⟨pw_56_0, hpre⟩ := hpre
  extract_lets -merge pw_57_0 at hpre
  obtain ⟨pw_59_0, hpre⟩ := hpre
  extract_lets -merge pw_60_0 pw_61_0 pw_62_0 pw_64_0 at hpre
  obtain ⟨pw_65_0, hpre⟩ := hpre
  extract_lets -merge pw_66_0 at hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  obtain ⟨pw_68_0, hpre⟩ := hpre
  extract_lets -merge pw_69_0 pw_70_0 at hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  obtain ⟨pw_75_0, hpre⟩ := hpre
  obtain ⟨pw_76_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_385.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_385.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_385.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_385.constraints_3] at hpre
  obtain ⟨-, -, -, ph1_0, -, -, -, ph2_0, ph3_0, ph5_0, -, -, -, ph7_0, -, -, -, ph8_0, ph9_0, ph11_0, -, -, -, -, -, -, ph14_0, -, -, -, -, -, -, ph15_0, ph16_0, -, -, ph17_0, ph18_0, ph19_0, ph22_0, ph25_0, -, -, -, -, -, -, ph28_0, -, -, -, -, -, -, ph29_0, ph30_0, ph32_0, ph33_0, ph34_0, ph42_0, ph46_0, ph48_0, ph49_0, -, -, -, -, -, -, ph52_0, ph53_0, -, -, ph54_0, ph55_0, ph56_0, ph59_0, ph65_0, ph67_0, ph68_0, -, -, -, -, -, -, ph71_0, ph72_0, -, -, -, -, -, -, ph73_0, ph74_0, -, -, -, -, -, -, ph75_0, ph76_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h4_0 h6_0 h11_0 h13_0
  obtain ⟨hk, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_4_0 := by
    rw [hashSample_eq ph1_0, hw4, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e2 : pw_2_0 = w_6_0 := by
    rw [hashSample_eq ph2_0, hw6, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e7 : pw_7_0 = w_11_0 := by
    rw [hashSample_eq ph7_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e8 : pw_8_0 = w_13_0 := by
    rw [hashSample_eq ph8_0, hw13, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The public multiplication by the one encoding.
  obtain ⟨t20, b20⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h18_0 h19_0 ph5_0 ph11_0
  have r20 := rebase_self (by inst_num) h20_0
  obtain ⟨t35, b35⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h33_0 h34_0 ph22_0 ph25_0
  have r35 := rebase_self (by inst_num) h35_0
  obtain ⟨-, s21, s21'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_20_0
  obtain ⟨-, s36, s36'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_35_0
  obtain ⟨p3a, p3b⟩ := stack2_mul_rows ph3_0 pw_5_0
  obtain ⟨p9a, p9b⟩ := stack2_mul_rows ph9_0 pw_11_0
  obtain ⟨p3a', p3b'⟩ := stack2_mul_rows ph3_0 pw_22_0
  obtain ⟨p9a', p9b'⟩ := stack2_mul_rows ph9_0 pw_25_0
  have k28 : w_28_0 = pw_14_0 := by
    rw [slice_rowOf (by inst_num) h28_0, slice_rowOf (by inst_num) ph14_0]
    show rowOf (w_15_0 * w_20_0) 1 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 0
    rw [s21, pairMul, rowOf_add, p3a, p9a, r20, t20, b20, e1, e7]
    rfl
  have k29 : w_29_0 = pw_15_0 := by
    rw [slice_rowOf (by inst_num) h29_0, slice_rowOf (by inst_num) ph15_0]
    show rowOf (w_15_0 * w_20_0) 2 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 1
    rw [s21', pairMul, rowOf_add, p3b, p9b, r20, t20, b20, e2, e8]
    rfl
  have k41 : w_41_0 = pw_28_0 := by
    rw [slice_rowOf (by inst_num) h41_0, slice_rowOf (by inst_num) ph28_0]
    show rowOf (w_15_0 * w_35_0) 1 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 0
    rw [s36, pairMul, rowOf_add, p3a', p9a', r35, t35, b35, e1, e7]
    rfl
  have k42 : w_42_0 = pw_29_0 := by
    rw [slice_rowOf (by inst_num) h42_0, slice_rowOf (by inst_num) ph29_0]
    show rowOf (w_15_0 * w_35_0) 2 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 1
    rw [s36', pairMul, rowOf_add, p3b', p9b', r35, t35, b35, e2, e8]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_46_0 = pw_17_0 := (hashSample_eq h46_0).trans (hashSample_eq ph17_0).symm
  have et1 : w_54_0 = pw_54_0 := (hashSample_eq h54_0).trans (hashSample_eq ph54_0).symm
  have eb0 : w_47_0 = pw_47_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_55_0 = pw_66_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h48_0
  rw [et1, eb1] at h56_0
  obtain ⟨t50, b50⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h48_0 h49_0 ph18_0 ph48_0
  have t50' := rebase_topRows h50_0 ph19_0 t50
  have b50' := rebase_bottomRows h50_0 ph49_0 b50
  obtain ⟨t58, b58⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h56_0 h57_0 ph55_0 ph67_0
  have t58' := rebase_topRows h58_0 ph56_0 t58
  have b58' := rebase_bottomRows h58_0 ph68_0 b58
  obtain ⟨-, s51, s51'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_50_0
  obtain ⟨-, s59, s59'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_58_0
  obtain ⟨q16a, q16b⟩ := stack2_mul_rows ph16_0 pw_19_0
  obtain ⟨q30a, q30b⟩ := stack2_mul_rows ph30_0 pw_49_0
  obtain ⟨q16a', q16b'⟩ := stack2_mul_rows ph16_0 pw_56_0
  obtain ⟨q30a', q30b'⟩ := stack2_mul_rows ph30_0 pw_68_0
  have k52 : w_52_0 = pw_52_0 := by
    rw [slice_rowOf (by inst_num) h52_0, slice_rowOf (by inst_num) ph52_0]
    show rowOf (w_44_0 * w_50_0) 1 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 0
    rw [s51, pairMul, rowOf_add, q16a, q30a, t50', b50', k28, k41]
    rfl
  have k62 : w_62_0 = pw_73_0 := by
    rw [slice_rowOf (by inst_num) h62_0, slice_rowOf (by inst_num) ph73_0]
    show rowOf (w_44_0 * w_50_0) 2 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 1
    rw [s51', pairMul, rowOf_add, q16b, q30b, t50', b50', k29, k42]
    rfl
  have k60 : w_60_0 = pw_71_0 := by
    rw [slice_rowOf (by inst_num) h60_0, slice_rowOf (by inst_num) ph71_0]
    show rowOf (w_44_0 * w_58_0) 1 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 0
    rw [s59, pairMul, rowOf_add, q16a', q30a', t58', b58', k28, k41]
    rfl
  have k64 : w_64_0 = pw_75_0 := by
    rw [slice_rowOf (by inst_num) h64_0, slice_rowOf (by inst_num) ph75_0]
    show rowOf (w_44_0 * w_58_0) 2 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 1
    rw [s59', pairMul, rowOf_add, q16b', q30b', t58', b58', k29, k42]
    rfl
  rw [k52] at h53_0
  rw [k62] at h63_0
  rw [k60] at h61_0
  rw [k64] at h65_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h53_0 ph53_0, blockModSwitchRuns_unique h61_0 ph72_0,
    blockModSwitchRuns_unique h63_0 ph74_0, blockModSwitchRuns_unique h65_0 ph76_0⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `1` of the smudge PRF (online `613`) are those the
preprocessing computes (`385`). -/
theorem keysCell613 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12} {apay : Fin 5 → ExactMatrix Q6 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : Fin 160 → ExactMatrix Q5 N 1 8}
    {bot0 bot1 : ExactMatrix Q0 N 1 6} {outs : Blocks Q0 6}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_615 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q6 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_389 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q6 N 1 12 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_615 at hon
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_5_0, hon⟩ := hon
  extract_lets -merge w_6_0 at hon
  obtain ⟨w_7_0, hon⟩ := hon
  extract_lets -merge w_8_0 at hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  extract_lets -merge w_15_0 at hon
  obtain ⟨w_16_0, hon⟩ := hon
  extract_lets -merge w_17_0 at hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_23_0, hon⟩ := hon
  obtain ⟨w_23_concat_1, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 at hon
  obtain ⟨w_27_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  extract_lets -merge w_30_0 at hon
  obtain ⟨w_31_0, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  extract_lets -merge w_34_0 at hon
  obtain ⟨w_35_0, hon⟩ := hon
  obtain ⟨w_36_0, hon⟩ := hon
  obtain ⟨w_36_concat_1, hon⟩ := hon
  obtain ⟨w_37_0, hon⟩ := hon
  obtain ⟨w_40_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_45_0, hon⟩ := hon
  extract_lets -merge w_46_0 at hon
  obtain ⟨w_47_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_51_0, hon⟩ := hon
  obtain ⟨w_52_0, hon⟩ := hon
  extract_lets -merge w_53_0 w_54_0 at hon
  obtain ⟨w_55_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_57_concat_1, hon⟩ := hon
  obtain ⟨w_59_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  extract_lets -merge w_65_0 at hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  extract_lets -merge w_69_0 w_70_0 at hon
  obtain ⟨w_71_0, hon⟩ := hon
  obtain ⟨w_72_0, hon⟩ := hon
  obtain ⟨w_73_0, hon⟩ := hon
  obtain ⟨w_73_concat_1, hon⟩ := hon
  obtain ⟨w_74_0, hon⟩ := hon
  obtain ⟨w_76_0, hon⟩ := hon
  obtain ⟨w_78_0, hon⟩ := hon
  obtain ⟨w_79_0, hon⟩ := hon
  obtain ⟨w_80_0, hon⟩ := hon
  extract_lets -merge w_81_0 at hon
  obtain ⟨w_82_0, hon⟩ := hon
  obtain ⟨w_83_0, hon⟩ := hon
  obtain ⟨w_84_0, hon⟩ := hon
  obtain ⟨w_86_0, hon⟩ := hon
  obtain ⟨w_87_0, hon⟩ := hon
  obtain ⟨w_88_0, hon⟩ := hon
  extract_lets -merge w_89_0 at hon
  obtain ⟨w_90_0, hon⟩ := hon
  obtain ⟨w_91_0, hon⟩ := hon
  obtain ⟨w_92_0, hon⟩ := hon
  obtain ⟨w_93_0, hon⟩ := hon
  obtain ⟨w_94_0, hon⟩ := hon
  obtain ⟨w_95_0, hon⟩ := hon
  obtain ⟨w_96_0, hon⟩ := hon
  obtain ⟨w_97_0, hon⟩ := hon
  obtain ⟨w_98_0, hon⟩ := hon
  obtain ⟨w_99_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_615.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_615.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_615.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_615.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_615.constraints_4, W_Stage_online_eval.parallel_parallel_generatedRoot_290_615.constraints_5] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, -, -, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, -, -, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_389 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_13_0 pw_25_0 pw_37_0 pw_41_0 pw_49_0 pw_56_0 pw_58_0 pw_76_0 pw_2_0 pw_14_0 pw_54_0 pw_57_0 pw_59_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_4_0, hpre⟩ := hpre
  extract_lets -merge pw_5_0 at hpre
  obtain ⟨pw_6_0, hpre⟩ := hpre
  extract_lets -merge pw_7_0 at hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  extract_lets -merge pw_10_0 at hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  obtain ⟨pw_12_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  extract_lets -merge pw_17_0 at hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  extract_lets -merge pw_19_0 at hpre
  obtain ⟨pw_20_0, hpre⟩ := hpre
  obtain ⟨pw_21_0, hpre⟩ := hpre
  extract_lets -merge pw_22_0 at hpre
  obtain ⟨pw_23_0, hpre⟩ := hpre
  obtain ⟨pw_24_0, hpre⟩ := hpre
  obtain ⟨pw_26_0, hpre⟩ := hpre
  obtain ⟨pw_27_0, hpre⟩ := hpre
  extract_lets -merge pw_28_0 pw_29_0 at hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_31_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  obtain ⟨pw_35_0, hpre⟩ := hpre
  extract_lets -merge pw_36_0 at hpre
  obtain ⟨pw_38_0, hpre⟩ := hpre
  obtain ⟨pw_39_0, hpre⟩ := hpre
  extract_lets -merge pw_40_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  obtain ⟨pw_43_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  obtain ⟨pw_47_0, hpre⟩ := hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_50_0, hpre⟩ := hpre
  extract_lets -merge pw_62_0 at hpre
  obtain ⟨pw_51_0, hpre⟩ := hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  extract_lets -merge pw_53_0 pw_55_0 at hpre
  obtain ⟨pw_60_0, hpre⟩ := hpre
  extract_lets -merge pw_61_0 pw_63_0 at hpre
  obtain ⟨pw_64_0, hpre⟩ := hpre
  extract_lets -merge pw_65_0 at hpre
  obtain ⟨pw_66_0, hpre⟩ := hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  extract_lets -merge pw_68_0 pw_69_0 at hpre
  obtain ⟨pw_70_0, hpre⟩ := hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  extract_lets -merge pw_81_0 at hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  extract_lets -merge pw_75_0 at hpre
  obtain ⟨pw_77_0, hpre⟩ := hpre
  extract_lets -merge pw_78_0 pw_79_0 pw_80_0 pw_82_0 at hpre
  obtain ⟨pw_83_0, hpre⟩ := hpre
  extract_lets -merge pw_84_0 at hpre
  obtain ⟨pw_85_0, hpre⟩ := hpre
  obtain ⟨pw_86_0, hpre⟩ := hpre
  extract_lets -merge pw_87_0 pw_88_0 at hpre
  obtain ⟨pw_89_0, hpre⟩ := hpre
  obtain ⟨pw_90_0, hpre⟩ := hpre
  obtain ⟨pw_91_0, hpre⟩ := hpre
  obtain ⟨pw_92_0, hpre⟩ := hpre
  obtain ⟨pw_93_0, hpre⟩ := hpre
  obtain ⟨pw_94_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_389.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_389.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_389.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_389.constraints_3] at hpre
  obtain ⟨-, -, -, ph1_0, ph3_0, ph4_0, ph6_0, ph8_0, -, -, -, ph9_0, ph11_0, ph12_0, -, ph14_0, ph14_0_1, ph15_0, ph16_0, -, -, -, ph18_0, ph20_0, -, -, -, ph21_0, ph23_0, ph24_0, ph14_0_2, ph14_0_3, ph26_0, ph27_0, -, -, -, -, -, -, ph30_0, -, -, -, -, -, -, ph31_0, ph32_0, -, -, ph33_0, ph34_0, ph35_0, ph14_0_4, ph14_0_5, ph38_0, ph39_0, ph14_0_6, ph14_0_7, ph42_0, ph43_0, -, -, -, -, -, -, ph46_0, -, -, -, -, -, -, ph47_0, ph48_0, ph50_0, ph51_0, ph52_0, -, -, ph60_0, ph64_0, ph66_0, ph67_0, -, -, -, -, -, -, ph70_0, ph71_0, -, -, ph72_0, ph73_0, ph74_0, ph77_0, ph83_0, ph85_0, ph86_0, -, -, -, -, -, -, ph89_0, ph90_0, -, -, -, -, -, -, ph91_0, ph92_0, -, -, -, -, -, -, ph93_0, ph94_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h11_0 h20_0 h29_0 h33_0
  obtain ⟨hk, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_11_0 := by
    rw [hashSample_eq ph1_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e9 : pw_9_0 = w_20_0 := by
    rw [hashSample_eq ph9_0, hw20, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e18 : pw_18_0 = w_29_0 := by
    rw [hashSample_eq ph18_0, hw29, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e21 : pw_21_0 = w_33_0 := by
    rw [hashSample_eq ph21_0, hw33, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The alignment `D(p G)` of the label.
  have g13 : w_13_0 = pw_3_0 := gadgetMatrixRuns_unique h13_0 ph3_0
  rw [g13] at h14_0
  have g14 : w_14_0 = pw_4_0 := centeredRebaseRuns_unique h14_0 ph4_0
  have g16 : w_16_0 = pw_6_0 := by
    apply gadgetDecomposeRuns_unique h16_0
    show gadgetDecomposeRuns _ _ _ (matrixMulScalarLeft _ w_14_0) _
    rw [g14]
    exact ph6_0
  have a18 : w_18_0 = pw_8_0 := by
    apply blockModSwitchRuns_unique h18_0
    show blockModSwitchRuns _ _ (w_11_0 * w_16_0) _
    rw [← e1, g16]
    exact ph8_0
  have a22 : w_22_0 = pw_11_0 := by
    apply blockModSwitchRuns_unique h22_0
    show blockModSwitchRuns _ _ (w_20_0 * w_16_0) _
    rw [← e9, g16]
    exact ph11_0
  have a31 : w_31_0 = pw_20_0 := by
    apply blockModSwitchRuns_unique h31_0
    show blockModSwitchRuns _ _ (w_29_0 * w_16_0) _
    rw [← e18, g16]
    exact ph20_0
  have a35 : w_35_0 = pw_23_0 := by
    apply blockModSwitchRuns_unique h35_0
    show blockModSwitchRuns _ _ (w_33_0 * w_16_0) _
    rw [← e21, g16]
    exact ph23_0
  -- The members of bucket `0`.
  have ho : w_39_0 = pw_14_0 := rfl
  rw [ho] at h40_0 h42_0 h59_0 h61_0
  have m40 : w_40_0 = pw_15_0 := familyGetDynamic_unique h40_0 ph15_0
  have m42 : w_42_0 = pw_26_0 := familyGetDynamic_unique h42_0 ph26_0
  have m59 : w_59_0 = pw_38_0 := familyGetDynamic_unique h59_0 ph38_0
  have m61 : w_61_0 = pw_42_0 := familyGetDynamic_unique h61_0 ph42_0
  rw [m40, m42] at h43_0
  rw [m59, m61] at h62_0
  obtain ⟨t45, b45⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h43_0 h44_0 ph16_0 ph27_0
  have r45 := rebase_self (by inst_num) h45_0
  obtain ⟨t64, b64⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h62_0 h63_0 ph39_0 ph43_0
  have r64 := rebase_self (by inst_num) h64_0
  obtain ⟨-, s46, s46'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_45_0
  obtain ⟨-, s65, s65'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_64_0
  obtain ⟨p12a, p12b⟩ := stack2_mul_rows ph12_0 pw_16_0
  obtain ⟨p24a, p24b⟩ := stack2_mul_rows ph24_0 pw_27_0
  obtain ⟨p12a', p12b'⟩ := stack2_mul_rows ph12_0 pw_39_0
  obtain ⟨p24a', p24b'⟩ := stack2_mul_rows ph24_0 pw_43_0
  have k55 : w_55_0 = pw_30_0 := by
    rw [slice_rowOf (by inst_num) h55_0, slice_rowOf (by inst_num) ph30_0]
    show rowOf (w_37_0 * w_45_0) 1 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 0
    rw [s46, pairMul, rowOf_add, p12a, p24a, r45, t45, b45, a18, a31]
    rfl
  have k56 : w_56_0 = pw_31_0 := by
    rw [slice_rowOf (by inst_num) h56_0, slice_rowOf (by inst_num) ph31_0]
    show rowOf (w_37_0 * w_45_0) 2 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 1
    rw [s46', pairMul, rowOf_add, p12b, p24b, r45, t45, b45, a22, a35]
    rfl
  have k71 : w_71_0 = pw_46_0 := by
    rw [slice_rowOf (by inst_num) h71_0, slice_rowOf (by inst_num) ph46_0]
    show rowOf (w_37_0 * w_64_0) 1 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 0
    rw [s65, pairMul, rowOf_add, p12a', p24a', r64, t64, b64, a18, a31]
    rfl
  have k72 : w_72_0 = pw_47_0 := by
    rw [slice_rowOf (by inst_num) h72_0, slice_rowOf (by inst_num) ph47_0]
    show rowOf (w_37_0 * w_64_0) 2 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 1
    rw [s65', pairMul, rowOf_add, p12b', p24b', r64, t64, b64, a22, a35]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_76_0 = pw_33_0 := (hashSample_eq h76_0).trans (hashSample_eq ph33_0).symm
  have et1 : w_84_0 = pw_72_0 := (hashSample_eq h84_0).trans (hashSample_eq ph72_0).symm
  have eb0 : w_77_0 = pw_65_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_85_0 = pw_84_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h78_0
  rw [et1, eb1] at h86_0
  obtain ⟨t80, b80⟩ := decompose_stack Backend.layout5 layoutQ0 rfl h78_0 h79_0 ph34_0 ph66_0
  have t80' := rebase_topRows h80_0 ph35_0 t80
  have b80' := rebase_bottomRows h80_0 ph67_0 b80
  obtain ⟨t88, b88⟩ := decompose_stack Backend.layout5 layoutQ0 rfl h86_0 h87_0 ph73_0 ph85_0
  have t88' := rebase_topRows h88_0 ph74_0 t88
  have b88' := rebase_bottomRows h88_0 ph86_0 b88
  obtain ⟨-, s81, s81'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_80_0
  obtain ⟨-, s89, s89'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_88_0
  obtain ⟨q32a, q32b⟩ := stack2_mul_rows ph32_0 pw_35_0
  obtain ⟨q48a, q48b⟩ := stack2_mul_rows ph48_0 pw_67_0
  obtain ⟨q32a', q32b'⟩ := stack2_mul_rows ph32_0 pw_74_0
  obtain ⟨q48a', q48b'⟩ := stack2_mul_rows ph48_0 pw_86_0
  have k82 : w_82_0 = pw_70_0 := by
    rw [slice_rowOf (by inst_num) h82_0, slice_rowOf (by inst_num) ph70_0]
    show rowOf (w_74_0 * w_80_0) 1 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 0
    rw [s81, pairMul, rowOf_add, q32a, q48a, t80', b80', k55, k71]
    rfl
  have k92 : w_92_0 = pw_91_0 := by
    rw [slice_rowOf (by inst_num) h92_0, slice_rowOf (by inst_num) ph91_0]
    show rowOf (w_74_0 * w_80_0) 2 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 1
    rw [s81', pairMul, rowOf_add, q32b, q48b, t80', b80', k56, k72]
    rfl
  have k90 : w_90_0 = pw_89_0 := by
    rw [slice_rowOf (by inst_num) h90_0, slice_rowOf (by inst_num) ph89_0]
    show rowOf (w_74_0 * w_88_0) 1 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 0
    rw [s89, pairMul, rowOf_add, q32a', q48a', t88', b88', k55, k71]
    rfl
  have k94 : w_94_0 = pw_93_0 := by
    rw [slice_rowOf (by inst_num) h94_0, slice_rowOf (by inst_num) ph93_0]
    show rowOf (w_74_0 * w_88_0) 2 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 1
    rw [s89', pairMul, rowOf_add, q32b', q48b', t88', b88', k56, k72]
    rfl
  rw [k82] at h83_0
  rw [k92] at h93_0
  rw [k90] at h91_0
  rw [k94] at h95_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h83_0 ph71_0, blockModSwitchRuns_unique h91_0 ph90_0,
    blockModSwitchRuns_unique h93_0 ph92_0, blockModSwitchRuns_unique h95_0 ph94_0⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `0` of the smudge PRF (online `663`) are those the
preprocessing computes (`425`). -/
theorem keysCell663 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12} {apay : Fin 5 → ExactMatrix Q6 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : ExactMatrix Q6 N 1 10} {bot0 bot1 : ExactMatrix Q5 N 1 8}
    {outs : Blocks Q5 8}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_665 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q6 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_429 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q6 N 1 12 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_665 at hon
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_6_0, hon⟩ := hon
  obtain ⟨w_7_0, hon⟩ := hon
  obtain ⟨w_7_concat_1, hon⟩ := hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  obtain ⟨w_14_concat_1, hon⟩ := hon
  obtain ⟨w_15_0, hon⟩ := hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_19_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hon
  obtain ⟨w_28_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  obtain ⟨w_30_0, hon⟩ := hon
  obtain ⟨w_30_concat_1, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  obtain ⟨w_34_0, hon⟩ := hon
  obtain ⟨w_35_0, hon⟩ := hon
  extract_lets -merge w_36_0 at hon
  obtain ⟨w_37_0, hon⟩ := hon
  extract_lets -merge w_40_0 at hon
  obtain ⟨w_41_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_43_concat_1, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_46_0, hon⟩ := hon
  obtain ⟨w_48_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_50_0, hon⟩ := hon
  extract_lets -merge w_51_0 at hon
  obtain ⟨w_52_0, hon⟩ := hon
  obtain ⟨w_53_0, hon⟩ := hon
  obtain ⟨w_54_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_58_0, hon⟩ := hon
  extract_lets -merge w_59_0 at hon
  obtain ⟨w_60_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  obtain ⟨w_65_0, hon⟩ := hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_67_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  obtain ⟨w_69_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_665.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_665.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_665.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_665.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_665.constraints_4] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, -, -, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, -, -, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_429 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_4_0 pw_10_0 pw_21_0 pw_24_0 pw_31_0 pw_38_0 pw_40_0 pw_58_0 pw_36_0 pw_39_0 pw_41_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_2_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_5_0, hpre⟩ := hpre
  extract_lets -merge pw_6_0 at hpre
  obtain ⟨pw_7_0, hpre⟩ := hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  extract_lets -merge pw_12_0 pw_13_0 at hpre
  obtain ⟨pw_14_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  obtain ⟨pw_17_0, hpre⟩ := hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  obtain ⟨pw_19_0, hpre⟩ := hpre
  extract_lets -merge pw_20_0 at hpre
  obtain ⟨pw_22_0, hpre⟩ := hpre
  extract_lets -merge pw_23_0 at hpre
  obtain ⟨pw_25_0, hpre⟩ := hpre
  extract_lets -merge pw_26_0 pw_27_0 at hpre
  obtain ⟨pw_28_0, hpre⟩ := hpre
  obtain ⟨pw_29_0, hpre⟩ := hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 at hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  extract_lets -merge pw_35_0 pw_37_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  extract_lets -merge pw_43_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  extract_lets -merge pw_47_0 at hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_49_0, hpre⟩ := hpre
  extract_lets -merge pw_50_0 pw_51_0 at hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  obtain ⟨pw_53_0, hpre⟩ := hpre
  obtain ⟨pw_54_0, hpre⟩ := hpre
  extract_lets -merge pw_63_0 at hpre
  obtain ⟨pw_55_0, hpre⟩ := hpre
  obtain ⟨pw_56_0, hpre⟩ := hpre
  extract_lets -merge pw_57_0 at hpre
  obtain ⟨pw_59_0, hpre⟩ := hpre
  extract_lets -merge pw_60_0 pw_61_0 pw_62_0 pw_64_0 at hpre
  obtain ⟨pw_65_0, hpre⟩ := hpre
  extract_lets -merge pw_66_0 at hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  obtain ⟨pw_68_0, hpre⟩ := hpre
  extract_lets -merge pw_69_0 pw_70_0 at hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  obtain ⟨pw_75_0, hpre⟩ := hpre
  obtain ⟨pw_76_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_429.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_429.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_429.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_429.constraints_3] at hpre
  obtain ⟨-, -, -, ph1_0, -, -, -, ph2_0, ph3_0, ph5_0, -, -, -, ph7_0, -, -, -, ph8_0, ph9_0, ph11_0, -, -, -, -, -, -, ph14_0, -, -, -, -, -, -, ph15_0, ph16_0, -, -, ph17_0, ph18_0, ph19_0, ph22_0, ph25_0, -, -, -, -, -, -, ph28_0, -, -, -, -, -, -, ph29_0, ph30_0, ph32_0, ph33_0, ph34_0, ph42_0, ph46_0, ph48_0, ph49_0, -, -, -, -, -, -, ph52_0, ph53_0, -, -, ph54_0, ph55_0, ph56_0, ph59_0, ph65_0, ph67_0, ph68_0, -, -, -, -, -, -, ph71_0, ph72_0, -, -, -, -, -, -, ph73_0, ph74_0, -, -, -, -, -, -, ph75_0, ph76_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h4_0 h6_0 h11_0 h13_0
  obtain ⟨hk, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_4_0 := by
    rw [hashSample_eq ph1_0, hw4, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e2 : pw_2_0 = w_6_0 := by
    rw [hashSample_eq ph2_0, hw6, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e7 : pw_7_0 = w_11_0 := by
    rw [hashSample_eq ph7_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e8 : pw_8_0 = w_13_0 := by
    rw [hashSample_eq ph8_0, hw13, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The public multiplication by the one encoding.
  obtain ⟨t20, b20⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h18_0 h19_0 ph5_0 ph11_0
  have r20 := rebase_self (by inst_num) h20_0
  obtain ⟨t35, b35⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h33_0 h34_0 ph22_0 ph25_0
  have r35 := rebase_self (by inst_num) h35_0
  obtain ⟨-, s21, s21'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_20_0
  obtain ⟨-, s36, s36'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_35_0
  obtain ⟨p3a, p3b⟩ := stack2_mul_rows ph3_0 pw_5_0
  obtain ⟨p9a, p9b⟩ := stack2_mul_rows ph9_0 pw_11_0
  obtain ⟨p3a', p3b'⟩ := stack2_mul_rows ph3_0 pw_22_0
  obtain ⟨p9a', p9b'⟩ := stack2_mul_rows ph9_0 pw_25_0
  have k28 : w_28_0 = pw_14_0 := by
    rw [slice_rowOf (by inst_num) h28_0, slice_rowOf (by inst_num) ph14_0]
    show rowOf (w_15_0 * w_20_0) 1 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 0
    rw [s21, pairMul, rowOf_add, p3a, p9a, r20, t20, b20, e1, e7]
    rfl
  have k29 : w_29_0 = pw_15_0 := by
    rw [slice_rowOf (by inst_num) h29_0, slice_rowOf (by inst_num) ph15_0]
    show rowOf (w_15_0 * w_20_0) 2 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 1
    rw [s21', pairMul, rowOf_add, p3b, p9b, r20, t20, b20, e2, e8]
    rfl
  have k41 : w_41_0 = pw_28_0 := by
    rw [slice_rowOf (by inst_num) h41_0, slice_rowOf (by inst_num) ph28_0]
    show rowOf (w_15_0 * w_35_0) 1 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 0
    rw [s36, pairMul, rowOf_add, p3a', p9a', r35, t35, b35, e1, e7]
    rfl
  have k42 : w_42_0 = pw_29_0 := by
    rw [slice_rowOf (by inst_num) h42_0, slice_rowOf (by inst_num) ph29_0]
    show rowOf (w_15_0 * w_35_0) 2 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 1
    rw [s36', pairMul, rowOf_add, p3b', p9b', r35, t35, b35, e2, e8]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_46_0 = pw_17_0 := (hashSample_eq h46_0).trans (hashSample_eq ph17_0).symm
  have et1 : w_54_0 = pw_54_0 := (hashSample_eq h54_0).trans (hashSample_eq ph54_0).symm
  have eb0 : w_47_0 = pw_47_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_55_0 = pw_66_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h48_0
  rw [et1, eb1] at h56_0
  obtain ⟨t50, b50⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h48_0 h49_0 ph18_0 ph48_0
  have t50' := rebase_topRows h50_0 ph19_0 t50
  have b50' := rebase_bottomRows h50_0 ph49_0 b50
  obtain ⟨t58, b58⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h56_0 h57_0 ph55_0 ph67_0
  have t58' := rebase_topRows h58_0 ph56_0 t58
  have b58' := rebase_bottomRows h58_0 ph68_0 b58
  obtain ⟨-, s51, s51'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_50_0
  obtain ⟨-, s59, s59'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_58_0
  obtain ⟨q16a, q16b⟩ := stack2_mul_rows ph16_0 pw_19_0
  obtain ⟨q30a, q30b⟩ := stack2_mul_rows ph30_0 pw_49_0
  obtain ⟨q16a', q16b'⟩ := stack2_mul_rows ph16_0 pw_56_0
  obtain ⟨q30a', q30b'⟩ := stack2_mul_rows ph30_0 pw_68_0
  have k52 : w_52_0 = pw_52_0 := by
    rw [slice_rowOf (by inst_num) h52_0, slice_rowOf (by inst_num) ph52_0]
    show rowOf (w_44_0 * w_50_0) 1 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 0
    rw [s51, pairMul, rowOf_add, q16a, q30a, t50', b50', k28, k41]
    rfl
  have k62 : w_62_0 = pw_73_0 := by
    rw [slice_rowOf (by inst_num) h62_0, slice_rowOf (by inst_num) ph73_0]
    show rowOf (w_44_0 * w_50_0) 2 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 1
    rw [s51', pairMul, rowOf_add, q16b, q30b, t50', b50', k29, k42]
    rfl
  have k60 : w_60_0 = pw_71_0 := by
    rw [slice_rowOf (by inst_num) h60_0, slice_rowOf (by inst_num) ph71_0]
    show rowOf (w_44_0 * w_58_0) 1 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 0
    rw [s59, pairMul, rowOf_add, q16a', q30a', t58', b58', k28, k41]
    rfl
  have k64 : w_64_0 = pw_75_0 := by
    rw [slice_rowOf (by inst_num) h64_0, slice_rowOf (by inst_num) ph75_0]
    show rowOf (w_44_0 * w_58_0) 2 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 1
    rw [s59', pairMul, rowOf_add, q16b', q30b', t58', b58', k29, k42]
    rfl
  rw [k52] at h53_0
  rw [k62] at h63_0
  rw [k60] at h61_0
  rw [k64] at h65_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h53_0 ph53_0, blockModSwitchRuns_unique h61_0 ph72_0,
    blockModSwitchRuns_unique h63_0 ph74_0, blockModSwitchRuns_unique h65_0 ph76_0⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `1` of the smudge PRF (online `667`) are those the
preprocessing computes (`429`). -/
theorem keysCell667 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12} {apay : Fin 5 → ExactMatrix Q6 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : Fin 160 → ExactMatrix Q5 N 1 8}
    {bot0 bot1 : ExactMatrix Q0 N 1 6} {outs : Blocks Q0 6}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_669 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q6 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_433 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q6 N 1 12 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_669 at hon
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_5_0, hon⟩ := hon
  extract_lets -merge w_6_0 at hon
  obtain ⟨w_7_0, hon⟩ := hon
  extract_lets -merge w_8_0 at hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  extract_lets -merge w_15_0 at hon
  obtain ⟨w_16_0, hon⟩ := hon
  extract_lets -merge w_17_0 at hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_23_0, hon⟩ := hon
  obtain ⟨w_23_concat_1, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 at hon
  obtain ⟨w_27_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  extract_lets -merge w_30_0 at hon
  obtain ⟨w_31_0, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  extract_lets -merge w_34_0 at hon
  obtain ⟨w_35_0, hon⟩ := hon
  obtain ⟨w_36_0, hon⟩ := hon
  obtain ⟨w_36_concat_1, hon⟩ := hon
  obtain ⟨w_37_0, hon⟩ := hon
  obtain ⟨w_40_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_45_0, hon⟩ := hon
  extract_lets -merge w_46_0 at hon
  obtain ⟨w_47_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_51_0, hon⟩ := hon
  obtain ⟨w_52_0, hon⟩ := hon
  extract_lets -merge w_53_0 w_54_0 at hon
  obtain ⟨w_55_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_57_concat_1, hon⟩ := hon
  obtain ⟨w_59_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  extract_lets -merge w_65_0 at hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  extract_lets -merge w_69_0 w_70_0 at hon
  obtain ⟨w_71_0, hon⟩ := hon
  obtain ⟨w_72_0, hon⟩ := hon
  obtain ⟨w_73_0, hon⟩ := hon
  obtain ⟨w_73_concat_1, hon⟩ := hon
  obtain ⟨w_74_0, hon⟩ := hon
  obtain ⟨w_76_0, hon⟩ := hon
  obtain ⟨w_78_0, hon⟩ := hon
  obtain ⟨w_79_0, hon⟩ := hon
  obtain ⟨w_80_0, hon⟩ := hon
  extract_lets -merge w_81_0 at hon
  obtain ⟨w_82_0, hon⟩ := hon
  obtain ⟨w_83_0, hon⟩ := hon
  obtain ⟨w_84_0, hon⟩ := hon
  obtain ⟨w_86_0, hon⟩ := hon
  obtain ⟨w_87_0, hon⟩ := hon
  obtain ⟨w_88_0, hon⟩ := hon
  extract_lets -merge w_89_0 at hon
  obtain ⟨w_90_0, hon⟩ := hon
  obtain ⟨w_91_0, hon⟩ := hon
  obtain ⟨w_92_0, hon⟩ := hon
  obtain ⟨w_93_0, hon⟩ := hon
  obtain ⟨w_94_0, hon⟩ := hon
  obtain ⟨w_95_0, hon⟩ := hon
  obtain ⟨w_96_0, hon⟩ := hon
  obtain ⟨w_97_0, hon⟩ := hon
  obtain ⟨w_98_0, hon⟩ := hon
  obtain ⟨w_99_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_669.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_669.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_669.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_669.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_669.constraints_4, W_Stage_online_eval.parallel_parallel_generatedRoot_290_669.constraints_5] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, -, -, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, -, -, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_433 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_13_0 pw_25_0 pw_37_0 pw_41_0 pw_49_0 pw_56_0 pw_58_0 pw_76_0 pw_2_0 pw_14_0 pw_54_0 pw_57_0 pw_59_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_4_0, hpre⟩ := hpre
  extract_lets -merge pw_5_0 at hpre
  obtain ⟨pw_6_0, hpre⟩ := hpre
  extract_lets -merge pw_7_0 at hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  extract_lets -merge pw_10_0 at hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  obtain ⟨pw_12_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  extract_lets -merge pw_17_0 at hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  extract_lets -merge pw_19_0 at hpre
  obtain ⟨pw_20_0, hpre⟩ := hpre
  obtain ⟨pw_21_0, hpre⟩ := hpre
  extract_lets -merge pw_22_0 at hpre
  obtain ⟨pw_23_0, hpre⟩ := hpre
  obtain ⟨pw_24_0, hpre⟩ := hpre
  obtain ⟨pw_26_0, hpre⟩ := hpre
  obtain ⟨pw_27_0, hpre⟩ := hpre
  extract_lets -merge pw_28_0 pw_29_0 at hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_31_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  obtain ⟨pw_35_0, hpre⟩ := hpre
  extract_lets -merge pw_36_0 at hpre
  obtain ⟨pw_38_0, hpre⟩ := hpre
  obtain ⟨pw_39_0, hpre⟩ := hpre
  extract_lets -merge pw_40_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  obtain ⟨pw_43_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  obtain ⟨pw_47_0, hpre⟩ := hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_50_0, hpre⟩ := hpre
  extract_lets -merge pw_62_0 at hpre
  obtain ⟨pw_51_0, hpre⟩ := hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  extract_lets -merge pw_53_0 pw_55_0 at hpre
  obtain ⟨pw_60_0, hpre⟩ := hpre
  extract_lets -merge pw_61_0 pw_63_0 at hpre
  obtain ⟨pw_64_0, hpre⟩ := hpre
  extract_lets -merge pw_65_0 at hpre
  obtain ⟨pw_66_0, hpre⟩ := hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  extract_lets -merge pw_68_0 pw_69_0 at hpre
  obtain ⟨pw_70_0, hpre⟩ := hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  extract_lets -merge pw_81_0 at hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  extract_lets -merge pw_75_0 at hpre
  obtain ⟨pw_77_0, hpre⟩ := hpre
  extract_lets -merge pw_78_0 pw_79_0 pw_80_0 pw_82_0 at hpre
  obtain ⟨pw_83_0, hpre⟩ := hpre
  extract_lets -merge pw_84_0 at hpre
  obtain ⟨pw_85_0, hpre⟩ := hpre
  obtain ⟨pw_86_0, hpre⟩ := hpre
  extract_lets -merge pw_87_0 pw_88_0 at hpre
  obtain ⟨pw_89_0, hpre⟩ := hpre
  obtain ⟨pw_90_0, hpre⟩ := hpre
  obtain ⟨pw_91_0, hpre⟩ := hpre
  obtain ⟨pw_92_0, hpre⟩ := hpre
  obtain ⟨pw_93_0, hpre⟩ := hpre
  obtain ⟨pw_94_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_433.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_433.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_433.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_433.constraints_3] at hpre
  obtain ⟨-, -, -, ph1_0, ph3_0, ph4_0, ph6_0, ph8_0, -, -, -, ph9_0, ph11_0, ph12_0, -, ph14_0, ph14_0_1, ph15_0, ph16_0, -, -, -, ph18_0, ph20_0, -, -, -, ph21_0, ph23_0, ph24_0, ph14_0_2, ph14_0_3, ph26_0, ph27_0, -, -, -, -, -, -, ph30_0, -, -, -, -, -, -, ph31_0, ph32_0, -, -, ph33_0, ph34_0, ph35_0, ph14_0_4, ph14_0_5, ph38_0, ph39_0, ph14_0_6, ph14_0_7, ph42_0, ph43_0, -, -, -, -, -, -, ph46_0, -, -, -, -, -, -, ph47_0, ph48_0, ph50_0, ph51_0, ph52_0, -, -, ph60_0, ph64_0, ph66_0, ph67_0, -, -, -, -, -, -, ph70_0, ph71_0, -, -, ph72_0, ph73_0, ph74_0, ph77_0, ph83_0, ph85_0, ph86_0, -, -, -, -, -, -, ph89_0, ph90_0, -, -, -, -, -, -, ph91_0, ph92_0, -, -, -, -, -, -, ph93_0, ph94_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h11_0 h20_0 h29_0 h33_0
  obtain ⟨hk, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_11_0 := by
    rw [hashSample_eq ph1_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e9 : pw_9_0 = w_20_0 := by
    rw [hashSample_eq ph9_0, hw20, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e18 : pw_18_0 = w_29_0 := by
    rw [hashSample_eq ph18_0, hw29, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e21 : pw_21_0 = w_33_0 := by
    rw [hashSample_eq ph21_0, hw33, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The alignment `D(p G)` of the label.
  have g13 : w_13_0 = pw_3_0 := gadgetMatrixRuns_unique h13_0 ph3_0
  rw [g13] at h14_0
  have g14 : w_14_0 = pw_4_0 := centeredRebaseRuns_unique h14_0 ph4_0
  have g16 : w_16_0 = pw_6_0 := by
    apply gadgetDecomposeRuns_unique h16_0
    show gadgetDecomposeRuns _ _ _ (matrixMulScalarLeft _ w_14_0) _
    rw [g14]
    exact ph6_0
  have a18 : w_18_0 = pw_8_0 := by
    apply blockModSwitchRuns_unique h18_0
    show blockModSwitchRuns _ _ (w_11_0 * w_16_0) _
    rw [← e1, g16]
    exact ph8_0
  have a22 : w_22_0 = pw_11_0 := by
    apply blockModSwitchRuns_unique h22_0
    show blockModSwitchRuns _ _ (w_20_0 * w_16_0) _
    rw [← e9, g16]
    exact ph11_0
  have a31 : w_31_0 = pw_20_0 := by
    apply blockModSwitchRuns_unique h31_0
    show blockModSwitchRuns _ _ (w_29_0 * w_16_0) _
    rw [← e18, g16]
    exact ph20_0
  have a35 : w_35_0 = pw_23_0 := by
    apply blockModSwitchRuns_unique h35_0
    show blockModSwitchRuns _ _ (w_33_0 * w_16_0) _
    rw [← e21, g16]
    exact ph23_0
  -- The members of bucket `0`.
  have ho : w_39_0 = pw_14_0 := rfl
  rw [ho] at h40_0 h42_0 h59_0 h61_0
  have m40 : w_40_0 = pw_15_0 := familyGetDynamic_unique h40_0 ph15_0
  have m42 : w_42_0 = pw_26_0 := familyGetDynamic_unique h42_0 ph26_0
  have m59 : w_59_0 = pw_38_0 := familyGetDynamic_unique h59_0 ph38_0
  have m61 : w_61_0 = pw_42_0 := familyGetDynamic_unique h61_0 ph42_0
  rw [m40, m42] at h43_0
  rw [m59, m61] at h62_0
  obtain ⟨t45, b45⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h43_0 h44_0 ph16_0 ph27_0
  have r45 := rebase_self (by inst_num) h45_0
  obtain ⟨t64, b64⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h62_0 h63_0 ph39_0 ph43_0
  have r64 := rebase_self (by inst_num) h64_0
  obtain ⟨-, s46, s46'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_45_0
  obtain ⟨-, s65, s65'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_64_0
  obtain ⟨p12a, p12b⟩ := stack2_mul_rows ph12_0 pw_16_0
  obtain ⟨p24a, p24b⟩ := stack2_mul_rows ph24_0 pw_27_0
  obtain ⟨p12a', p12b'⟩ := stack2_mul_rows ph12_0 pw_39_0
  obtain ⟨p24a', p24b'⟩ := stack2_mul_rows ph24_0 pw_43_0
  have k55 : w_55_0 = pw_30_0 := by
    rw [slice_rowOf (by inst_num) h55_0, slice_rowOf (by inst_num) ph30_0]
    show rowOf (w_37_0 * w_45_0) 1 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 0
    rw [s46, pairMul, rowOf_add, p12a, p24a, r45, t45, b45, a18, a31]
    rfl
  have k56 : w_56_0 = pw_31_0 := by
    rw [slice_rowOf (by inst_num) h56_0, slice_rowOf (by inst_num) ph31_0]
    show rowOf (w_37_0 * w_45_0) 2 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 1
    rw [s46', pairMul, rowOf_add, p12b, p24b, r45, t45, b45, a22, a35]
    rfl
  have k71 : w_71_0 = pw_46_0 := by
    rw [slice_rowOf (by inst_num) h71_0, slice_rowOf (by inst_num) ph46_0]
    show rowOf (w_37_0 * w_64_0) 1 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 0
    rw [s65, pairMul, rowOf_add, p12a', p24a', r64, t64, b64, a18, a31]
    rfl
  have k72 : w_72_0 = pw_47_0 := by
    rw [slice_rowOf (by inst_num) h72_0, slice_rowOf (by inst_num) ph47_0]
    show rowOf (w_37_0 * w_64_0) 2 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 1
    rw [s65', pairMul, rowOf_add, p12b', p24b', r64, t64, b64, a22, a35]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_76_0 = pw_33_0 := (hashSample_eq h76_0).trans (hashSample_eq ph33_0).symm
  have et1 : w_84_0 = pw_72_0 := (hashSample_eq h84_0).trans (hashSample_eq ph72_0).symm
  have eb0 : w_77_0 = pw_65_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_85_0 = pw_84_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h78_0
  rw [et1, eb1] at h86_0
  obtain ⟨t80, b80⟩ := decompose_stack Backend.layout5 layoutQ0 rfl h78_0 h79_0 ph34_0 ph66_0
  have t80' := rebase_topRows h80_0 ph35_0 t80
  have b80' := rebase_bottomRows h80_0 ph67_0 b80
  obtain ⟨t88, b88⟩ := decompose_stack Backend.layout5 layoutQ0 rfl h86_0 h87_0 ph73_0 ph85_0
  have t88' := rebase_topRows h88_0 ph74_0 t88
  have b88' := rebase_bottomRows h88_0 ph86_0 b88
  obtain ⟨-, s81, s81'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_80_0
  obtain ⟨-, s89, s89'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_88_0
  obtain ⟨q32a, q32b⟩ := stack2_mul_rows ph32_0 pw_35_0
  obtain ⟨q48a, q48b⟩ := stack2_mul_rows ph48_0 pw_67_0
  obtain ⟨q32a', q32b'⟩ := stack2_mul_rows ph32_0 pw_74_0
  obtain ⟨q48a', q48b'⟩ := stack2_mul_rows ph48_0 pw_86_0
  have k82 : w_82_0 = pw_70_0 := by
    rw [slice_rowOf (by inst_num) h82_0, slice_rowOf (by inst_num) ph70_0]
    show rowOf (w_74_0 * w_80_0) 1 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 0
    rw [s81, pairMul, rowOf_add, q32a, q48a, t80', b80', k55, k71]
    rfl
  have k92 : w_92_0 = pw_91_0 := by
    rw [slice_rowOf (by inst_num) h92_0, slice_rowOf (by inst_num) ph91_0]
    show rowOf (w_74_0 * w_80_0) 2 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 1
    rw [s81', pairMul, rowOf_add, q32b, q48b, t80', b80', k56, k72]
    rfl
  have k90 : w_90_0 = pw_89_0 := by
    rw [slice_rowOf (by inst_num) h90_0, slice_rowOf (by inst_num) ph89_0]
    show rowOf (w_74_0 * w_88_0) 1 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 0
    rw [s89, pairMul, rowOf_add, q32a', q48a', t88', b88', k55, k71]
    rfl
  have k94 : w_94_0 = pw_93_0 := by
    rw [slice_rowOf (by inst_num) h94_0, slice_rowOf (by inst_num) ph93_0]
    show rowOf (w_74_0 * w_88_0) 2 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 1
    rw [s89', pairMul, rowOf_add, q32b', q48b', t88', b88', k56, k72]
    rfl
  rw [k82] at h83_0
  rw [k92] at h93_0
  rw [k90] at h91_0
  rw [k94] at h95_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h83_0 ph71_0, blockModSwitchRuns_unique h91_0 ph90_0,
    blockModSwitchRuns_unique h93_0 ph92_0, blockModSwitchRuns_unique h95_0 ph94_0⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `0` of the smudge PRF (online `717`) are those the
preprocessing computes (`469`). -/
theorem keysCell717 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12} {apay : Fin 5 → ExactMatrix Q6 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : ExactMatrix Q6 N 1 10} {bot0 bot1 : ExactMatrix Q5 N 1 8}
    {outs : Blocks Q5 8}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_719 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q6 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_473 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q6 N 1 12 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_719 at hon
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_6_0, hon⟩ := hon
  obtain ⟨w_7_0, hon⟩ := hon
  obtain ⟨w_7_concat_1, hon⟩ := hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  obtain ⟨w_14_concat_1, hon⟩ := hon
  obtain ⟨w_15_0, hon⟩ := hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_19_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hon
  obtain ⟨w_28_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  obtain ⟨w_30_0, hon⟩ := hon
  obtain ⟨w_30_concat_1, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  obtain ⟨w_34_0, hon⟩ := hon
  obtain ⟨w_35_0, hon⟩ := hon
  extract_lets -merge w_36_0 at hon
  obtain ⟨w_37_0, hon⟩ := hon
  extract_lets -merge w_40_0 at hon
  obtain ⟨w_41_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_43_concat_1, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_46_0, hon⟩ := hon
  obtain ⟨w_48_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_50_0, hon⟩ := hon
  extract_lets -merge w_51_0 at hon
  obtain ⟨w_52_0, hon⟩ := hon
  obtain ⟨w_53_0, hon⟩ := hon
  obtain ⟨w_54_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_58_0, hon⟩ := hon
  extract_lets -merge w_59_0 at hon
  obtain ⟨w_60_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  obtain ⟨w_65_0, hon⟩ := hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_67_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  obtain ⟨w_69_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_719.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_719.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_719.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_719.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_719.constraints_4] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, -, -, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, -, -, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_473 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_4_0 pw_10_0 pw_21_0 pw_24_0 pw_31_0 pw_38_0 pw_40_0 pw_58_0 pw_36_0 pw_39_0 pw_41_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_2_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_5_0, hpre⟩ := hpre
  extract_lets -merge pw_6_0 at hpre
  obtain ⟨pw_7_0, hpre⟩ := hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  extract_lets -merge pw_12_0 pw_13_0 at hpre
  obtain ⟨pw_14_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  obtain ⟨pw_17_0, hpre⟩ := hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  obtain ⟨pw_19_0, hpre⟩ := hpre
  extract_lets -merge pw_20_0 at hpre
  obtain ⟨pw_22_0, hpre⟩ := hpre
  extract_lets -merge pw_23_0 at hpre
  obtain ⟨pw_25_0, hpre⟩ := hpre
  extract_lets -merge pw_26_0 pw_27_0 at hpre
  obtain ⟨pw_28_0, hpre⟩ := hpre
  obtain ⟨pw_29_0, hpre⟩ := hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 at hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  extract_lets -merge pw_35_0 pw_37_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  extract_lets -merge pw_43_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  extract_lets -merge pw_47_0 at hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_49_0, hpre⟩ := hpre
  extract_lets -merge pw_50_0 pw_51_0 at hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  obtain ⟨pw_53_0, hpre⟩ := hpre
  obtain ⟨pw_54_0, hpre⟩ := hpre
  extract_lets -merge pw_63_0 at hpre
  obtain ⟨pw_55_0, hpre⟩ := hpre
  obtain ⟨pw_56_0, hpre⟩ := hpre
  extract_lets -merge pw_57_0 at hpre
  obtain ⟨pw_59_0, hpre⟩ := hpre
  extract_lets -merge pw_60_0 pw_61_0 pw_62_0 pw_64_0 at hpre
  obtain ⟨pw_65_0, hpre⟩ := hpre
  extract_lets -merge pw_66_0 at hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  obtain ⟨pw_68_0, hpre⟩ := hpre
  extract_lets -merge pw_69_0 pw_70_0 at hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  obtain ⟨pw_75_0, hpre⟩ := hpre
  obtain ⟨pw_76_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_473.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_473.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_473.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_473.constraints_3] at hpre
  obtain ⟨-, -, -, ph1_0, -, -, -, ph2_0, ph3_0, ph5_0, -, -, -, ph7_0, -, -, -, ph8_0, ph9_0, ph11_0, -, -, -, -, -, -, ph14_0, -, -, -, -, -, -, ph15_0, ph16_0, -, -, ph17_0, ph18_0, ph19_0, ph22_0, ph25_0, -, -, -, -, -, -, ph28_0, -, -, -, -, -, -, ph29_0, ph30_0, ph32_0, ph33_0, ph34_0, ph42_0, ph46_0, ph48_0, ph49_0, -, -, -, -, -, -, ph52_0, ph53_0, -, -, ph54_0, ph55_0, ph56_0, ph59_0, ph65_0, ph67_0, ph68_0, -, -, -, -, -, -, ph71_0, ph72_0, -, -, -, -, -, -, ph73_0, ph74_0, -, -, -, -, -, -, ph75_0, ph76_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h4_0 h6_0 h11_0 h13_0
  obtain ⟨hk, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_4_0 := by
    rw [hashSample_eq ph1_0, hw4, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e2 : pw_2_0 = w_6_0 := by
    rw [hashSample_eq ph2_0, hw6, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e7 : pw_7_0 = w_11_0 := by
    rw [hashSample_eq ph7_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e8 : pw_8_0 = w_13_0 := by
    rw [hashSample_eq ph8_0, hw13, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The public multiplication by the one encoding.
  obtain ⟨t20, b20⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h18_0 h19_0 ph5_0 ph11_0
  have r20 := rebase_self (by inst_num) h20_0
  obtain ⟨t35, b35⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h33_0 h34_0 ph22_0 ph25_0
  have r35 := rebase_self (by inst_num) h35_0
  obtain ⟨-, s21, s21'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_20_0
  obtain ⟨-, s36, s36'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_35_0
  obtain ⟨p3a, p3b⟩ := stack2_mul_rows ph3_0 pw_5_0
  obtain ⟨p9a, p9b⟩ := stack2_mul_rows ph9_0 pw_11_0
  obtain ⟨p3a', p3b'⟩ := stack2_mul_rows ph3_0 pw_22_0
  obtain ⟨p9a', p9b'⟩ := stack2_mul_rows ph9_0 pw_25_0
  have k28 : w_28_0 = pw_14_0 := by
    rw [slice_rowOf (by inst_num) h28_0, slice_rowOf (by inst_num) ph14_0]
    show rowOf (w_15_0 * w_20_0) 1 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 0
    rw [s21, pairMul, rowOf_add, p3a, p9a, r20, t20, b20, e1, e7]
    rfl
  have k29 : w_29_0 = pw_15_0 := by
    rw [slice_rowOf (by inst_num) h29_0, slice_rowOf (by inst_num) ph15_0]
    show rowOf (w_15_0 * w_20_0) 2 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 1
    rw [s21', pairMul, rowOf_add, p3b, p9b, r20, t20, b20, e2, e8]
    rfl
  have k41 : w_41_0 = pw_28_0 := by
    rw [slice_rowOf (by inst_num) h41_0, slice_rowOf (by inst_num) ph28_0]
    show rowOf (w_15_0 * w_35_0) 1 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 0
    rw [s36, pairMul, rowOf_add, p3a', p9a', r35, t35, b35, e1, e7]
    rfl
  have k42 : w_42_0 = pw_29_0 := by
    rw [slice_rowOf (by inst_num) h42_0, slice_rowOf (by inst_num) ph29_0]
    show rowOf (w_15_0 * w_35_0) 2 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 1
    rw [s36', pairMul, rowOf_add, p3b', p9b', r35, t35, b35, e2, e8]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_46_0 = pw_17_0 := (hashSample_eq h46_0).trans (hashSample_eq ph17_0).symm
  have et1 : w_54_0 = pw_54_0 := (hashSample_eq h54_0).trans (hashSample_eq ph54_0).symm
  have eb0 : w_47_0 = pw_47_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_55_0 = pw_66_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h48_0
  rw [et1, eb1] at h56_0
  obtain ⟨t50, b50⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h48_0 h49_0 ph18_0 ph48_0
  have t50' := rebase_topRows h50_0 ph19_0 t50
  have b50' := rebase_bottomRows h50_0 ph49_0 b50
  obtain ⟨t58, b58⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h56_0 h57_0 ph55_0 ph67_0
  have t58' := rebase_topRows h58_0 ph56_0 t58
  have b58' := rebase_bottomRows h58_0 ph68_0 b58
  obtain ⟨-, s51, s51'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_50_0
  obtain ⟨-, s59, s59'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_58_0
  obtain ⟨q16a, q16b⟩ := stack2_mul_rows ph16_0 pw_19_0
  obtain ⟨q30a, q30b⟩ := stack2_mul_rows ph30_0 pw_49_0
  obtain ⟨q16a', q16b'⟩ := stack2_mul_rows ph16_0 pw_56_0
  obtain ⟨q30a', q30b'⟩ := stack2_mul_rows ph30_0 pw_68_0
  have k52 : w_52_0 = pw_52_0 := by
    rw [slice_rowOf (by inst_num) h52_0, slice_rowOf (by inst_num) ph52_0]
    show rowOf (w_44_0 * w_50_0) 1 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 0
    rw [s51, pairMul, rowOf_add, q16a, q30a, t50', b50', k28, k41]
    rfl
  have k62 : w_62_0 = pw_73_0 := by
    rw [slice_rowOf (by inst_num) h62_0, slice_rowOf (by inst_num) ph73_0]
    show rowOf (w_44_0 * w_50_0) 2 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 1
    rw [s51', pairMul, rowOf_add, q16b, q30b, t50', b50', k29, k42]
    rfl
  have k60 : w_60_0 = pw_71_0 := by
    rw [slice_rowOf (by inst_num) h60_0, slice_rowOf (by inst_num) ph71_0]
    show rowOf (w_44_0 * w_58_0) 1 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 0
    rw [s59, pairMul, rowOf_add, q16a', q30a', t58', b58', k28, k41]
    rfl
  have k64 : w_64_0 = pw_75_0 := by
    rw [slice_rowOf (by inst_num) h64_0, slice_rowOf (by inst_num) ph75_0]
    show rowOf (w_44_0 * w_58_0) 2 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 1
    rw [s59', pairMul, rowOf_add, q16b', q30b', t58', b58', k29, k42]
    rfl
  rw [k52] at h53_0
  rw [k62] at h63_0
  rw [k60] at h61_0
  rw [k64] at h65_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h53_0 ph53_0, blockModSwitchRuns_unique h61_0 ph72_0,
    blockModSwitchRuns_unique h63_0 ph74_0, blockModSwitchRuns_unique h65_0 ph76_0⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `1` of the smudge PRF (online `721`) are those the
preprocessing computes (`473`). -/
theorem keysCell721 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12} {apay : Fin 5 → ExactMatrix Q6 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : Fin 160 → ExactMatrix Q5 N 1 8}
    {bot0 bot1 : ExactMatrix Q0 N 1 6} {outs : Blocks Q0 6}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_723 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q6 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_477 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q6 N 1 12 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_723 at hon
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_5_0, hon⟩ := hon
  extract_lets -merge w_6_0 at hon
  obtain ⟨w_7_0, hon⟩ := hon
  extract_lets -merge w_8_0 at hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  extract_lets -merge w_15_0 at hon
  obtain ⟨w_16_0, hon⟩ := hon
  extract_lets -merge w_17_0 at hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_23_0, hon⟩ := hon
  obtain ⟨w_23_concat_1, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 at hon
  obtain ⟨w_27_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  extract_lets -merge w_30_0 at hon
  obtain ⟨w_31_0, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  extract_lets -merge w_34_0 at hon
  obtain ⟨w_35_0, hon⟩ := hon
  obtain ⟨w_36_0, hon⟩ := hon
  obtain ⟨w_36_concat_1, hon⟩ := hon
  obtain ⟨w_37_0, hon⟩ := hon
  obtain ⟨w_40_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_45_0, hon⟩ := hon
  extract_lets -merge w_46_0 at hon
  obtain ⟨w_47_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_51_0, hon⟩ := hon
  obtain ⟨w_52_0, hon⟩ := hon
  extract_lets -merge w_53_0 w_54_0 at hon
  obtain ⟨w_55_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_57_concat_1, hon⟩ := hon
  obtain ⟨w_59_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  extract_lets -merge w_65_0 at hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  extract_lets -merge w_69_0 w_70_0 at hon
  obtain ⟨w_71_0, hon⟩ := hon
  obtain ⟨w_72_0, hon⟩ := hon
  obtain ⟨w_73_0, hon⟩ := hon
  obtain ⟨w_73_concat_1, hon⟩ := hon
  obtain ⟨w_74_0, hon⟩ := hon
  obtain ⟨w_76_0, hon⟩ := hon
  obtain ⟨w_78_0, hon⟩ := hon
  obtain ⟨w_79_0, hon⟩ := hon
  obtain ⟨w_80_0, hon⟩ := hon
  extract_lets -merge w_81_0 at hon
  obtain ⟨w_82_0, hon⟩ := hon
  obtain ⟨w_83_0, hon⟩ := hon
  obtain ⟨w_84_0, hon⟩ := hon
  obtain ⟨w_86_0, hon⟩ := hon
  obtain ⟨w_87_0, hon⟩ := hon
  obtain ⟨w_88_0, hon⟩ := hon
  extract_lets -merge w_89_0 at hon
  obtain ⟨w_90_0, hon⟩ := hon
  obtain ⟨w_91_0, hon⟩ := hon
  obtain ⟨w_92_0, hon⟩ := hon
  obtain ⟨w_93_0, hon⟩ := hon
  obtain ⟨w_94_0, hon⟩ := hon
  obtain ⟨w_95_0, hon⟩ := hon
  obtain ⟨w_96_0, hon⟩ := hon
  obtain ⟨w_97_0, hon⟩ := hon
  obtain ⟨w_98_0, hon⟩ := hon
  obtain ⟨w_99_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_723.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_723.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_723.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_723.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_723.constraints_4, W_Stage_online_eval.parallel_parallel_generatedRoot_290_723.constraints_5] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, -, -, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, -, -, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_477 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_13_0 pw_25_0 pw_37_0 pw_41_0 pw_49_0 pw_56_0 pw_58_0 pw_76_0 pw_2_0 pw_14_0 pw_54_0 pw_57_0 pw_59_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_4_0, hpre⟩ := hpre
  extract_lets -merge pw_5_0 at hpre
  obtain ⟨pw_6_0, hpre⟩ := hpre
  extract_lets -merge pw_7_0 at hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  extract_lets -merge pw_10_0 at hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  obtain ⟨pw_12_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  extract_lets -merge pw_17_0 at hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  extract_lets -merge pw_19_0 at hpre
  obtain ⟨pw_20_0, hpre⟩ := hpre
  obtain ⟨pw_21_0, hpre⟩ := hpre
  extract_lets -merge pw_22_0 at hpre
  obtain ⟨pw_23_0, hpre⟩ := hpre
  obtain ⟨pw_24_0, hpre⟩ := hpre
  obtain ⟨pw_26_0, hpre⟩ := hpre
  obtain ⟨pw_27_0, hpre⟩ := hpre
  extract_lets -merge pw_28_0 pw_29_0 at hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_31_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  obtain ⟨pw_35_0, hpre⟩ := hpre
  extract_lets -merge pw_36_0 at hpre
  obtain ⟨pw_38_0, hpre⟩ := hpre
  obtain ⟨pw_39_0, hpre⟩ := hpre
  extract_lets -merge pw_40_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  obtain ⟨pw_43_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  obtain ⟨pw_47_0, hpre⟩ := hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_50_0, hpre⟩ := hpre
  extract_lets -merge pw_62_0 at hpre
  obtain ⟨pw_51_0, hpre⟩ := hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  extract_lets -merge pw_53_0 pw_55_0 at hpre
  obtain ⟨pw_60_0, hpre⟩ := hpre
  extract_lets -merge pw_61_0 pw_63_0 at hpre
  obtain ⟨pw_64_0, hpre⟩ := hpre
  extract_lets -merge pw_65_0 at hpre
  obtain ⟨pw_66_0, hpre⟩ := hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  extract_lets -merge pw_68_0 pw_69_0 at hpre
  obtain ⟨pw_70_0, hpre⟩ := hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  extract_lets -merge pw_81_0 at hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  extract_lets -merge pw_75_0 at hpre
  obtain ⟨pw_77_0, hpre⟩ := hpre
  extract_lets -merge pw_78_0 pw_79_0 pw_80_0 pw_82_0 at hpre
  obtain ⟨pw_83_0, hpre⟩ := hpre
  extract_lets -merge pw_84_0 at hpre
  obtain ⟨pw_85_0, hpre⟩ := hpre
  obtain ⟨pw_86_0, hpre⟩ := hpre
  extract_lets -merge pw_87_0 pw_88_0 at hpre
  obtain ⟨pw_89_0, hpre⟩ := hpre
  obtain ⟨pw_90_0, hpre⟩ := hpre
  obtain ⟨pw_91_0, hpre⟩ := hpre
  obtain ⟨pw_92_0, hpre⟩ := hpre
  obtain ⟨pw_93_0, hpre⟩ := hpre
  obtain ⟨pw_94_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_477.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_477.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_477.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_477.constraints_3] at hpre
  obtain ⟨-, -, -, ph1_0, ph3_0, ph4_0, ph6_0, ph8_0, -, -, -, ph9_0, ph11_0, ph12_0, -, ph14_0, ph14_0_1, ph15_0, ph16_0, -, -, -, ph18_0, ph20_0, -, -, -, ph21_0, ph23_0, ph24_0, ph14_0_2, ph14_0_3, ph26_0, ph27_0, -, -, -, -, -, -, ph30_0, -, -, -, -, -, -, ph31_0, ph32_0, -, -, ph33_0, ph34_0, ph35_0, ph14_0_4, ph14_0_5, ph38_0, ph39_0, ph14_0_6, ph14_0_7, ph42_0, ph43_0, -, -, -, -, -, -, ph46_0, -, -, -, -, -, -, ph47_0, ph48_0, ph50_0, ph51_0, ph52_0, -, -, ph60_0, ph64_0, ph66_0, ph67_0, -, -, -, -, -, -, ph70_0, ph71_0, -, -, ph72_0, ph73_0, ph74_0, ph77_0, ph83_0, ph85_0, ph86_0, -, -, -, -, -, -, ph89_0, ph90_0, -, -, -, -, -, -, ph91_0, ph92_0, -, -, -, -, -, -, ph93_0, ph94_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h11_0 h20_0 h29_0 h33_0
  obtain ⟨hk, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_11_0 := by
    rw [hashSample_eq ph1_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e9 : pw_9_0 = w_20_0 := by
    rw [hashSample_eq ph9_0, hw20, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e18 : pw_18_0 = w_29_0 := by
    rw [hashSample_eq ph18_0, hw29, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e21 : pw_21_0 = w_33_0 := by
    rw [hashSample_eq ph21_0, hw33, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The alignment `D(p G)` of the label.
  have g13 : w_13_0 = pw_3_0 := gadgetMatrixRuns_unique h13_0 ph3_0
  rw [g13] at h14_0
  have g14 : w_14_0 = pw_4_0 := centeredRebaseRuns_unique h14_0 ph4_0
  have g16 : w_16_0 = pw_6_0 := by
    apply gadgetDecomposeRuns_unique h16_0
    show gadgetDecomposeRuns _ _ _ (matrixMulScalarLeft _ w_14_0) _
    rw [g14]
    exact ph6_0
  have a18 : w_18_0 = pw_8_0 := by
    apply blockModSwitchRuns_unique h18_0
    show blockModSwitchRuns _ _ (w_11_0 * w_16_0) _
    rw [← e1, g16]
    exact ph8_0
  have a22 : w_22_0 = pw_11_0 := by
    apply blockModSwitchRuns_unique h22_0
    show blockModSwitchRuns _ _ (w_20_0 * w_16_0) _
    rw [← e9, g16]
    exact ph11_0
  have a31 : w_31_0 = pw_20_0 := by
    apply blockModSwitchRuns_unique h31_0
    show blockModSwitchRuns _ _ (w_29_0 * w_16_0) _
    rw [← e18, g16]
    exact ph20_0
  have a35 : w_35_0 = pw_23_0 := by
    apply blockModSwitchRuns_unique h35_0
    show blockModSwitchRuns _ _ (w_33_0 * w_16_0) _
    rw [← e21, g16]
    exact ph23_0
  -- The members of bucket `0`.
  have ho : w_39_0 = pw_14_0 := rfl
  rw [ho] at h40_0 h42_0 h59_0 h61_0
  have m40 : w_40_0 = pw_15_0 := familyGetDynamic_unique h40_0 ph15_0
  have m42 : w_42_0 = pw_26_0 := familyGetDynamic_unique h42_0 ph26_0
  have m59 : w_59_0 = pw_38_0 := familyGetDynamic_unique h59_0 ph38_0
  have m61 : w_61_0 = pw_42_0 := familyGetDynamic_unique h61_0 ph42_0
  rw [m40, m42] at h43_0
  rw [m59, m61] at h62_0
  obtain ⟨t45, b45⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h43_0 h44_0 ph16_0 ph27_0
  have r45 := rebase_self (by inst_num) h45_0
  obtain ⟨t64, b64⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h62_0 h63_0 ph39_0 ph43_0
  have r64 := rebase_self (by inst_num) h64_0
  obtain ⟨-, s46, s46'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_45_0
  obtain ⟨-, s65, s65'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_64_0
  obtain ⟨p12a, p12b⟩ := stack2_mul_rows ph12_0 pw_16_0
  obtain ⟨p24a, p24b⟩ := stack2_mul_rows ph24_0 pw_27_0
  obtain ⟨p12a', p12b'⟩ := stack2_mul_rows ph12_0 pw_39_0
  obtain ⟨p24a', p24b'⟩ := stack2_mul_rows ph24_0 pw_43_0
  have k55 : w_55_0 = pw_30_0 := by
    rw [slice_rowOf (by inst_num) h55_0, slice_rowOf (by inst_num) ph30_0]
    show rowOf (w_37_0 * w_45_0) 1 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 0
    rw [s46, pairMul, rowOf_add, p12a, p24a, r45, t45, b45, a18, a31]
    rfl
  have k56 : w_56_0 = pw_31_0 := by
    rw [slice_rowOf (by inst_num) h56_0, slice_rowOf (by inst_num) ph31_0]
    show rowOf (w_37_0 * w_45_0) 2 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 1
    rw [s46', pairMul, rowOf_add, p12b, p24b, r45, t45, b45, a22, a35]
    rfl
  have k71 : w_71_0 = pw_46_0 := by
    rw [slice_rowOf (by inst_num) h71_0, slice_rowOf (by inst_num) ph46_0]
    show rowOf (w_37_0 * w_64_0) 1 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 0
    rw [s65, pairMul, rowOf_add, p12a', p24a', r64, t64, b64, a18, a31]
    rfl
  have k72 : w_72_0 = pw_47_0 := by
    rw [slice_rowOf (by inst_num) h72_0, slice_rowOf (by inst_num) ph47_0]
    show rowOf (w_37_0 * w_64_0) 2 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 1
    rw [s65', pairMul, rowOf_add, p12b', p24b', r64, t64, b64, a22, a35]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_76_0 = pw_33_0 := (hashSample_eq h76_0).trans (hashSample_eq ph33_0).symm
  have et1 : w_84_0 = pw_72_0 := (hashSample_eq h84_0).trans (hashSample_eq ph72_0).symm
  have eb0 : w_77_0 = pw_65_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_85_0 = pw_84_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h78_0
  rw [et1, eb1] at h86_0
  obtain ⟨t80, b80⟩ := decompose_stack Backend.layout5 layoutQ0 rfl h78_0 h79_0 ph34_0 ph66_0
  have t80' := rebase_topRows h80_0 ph35_0 t80
  have b80' := rebase_bottomRows h80_0 ph67_0 b80
  obtain ⟨t88, b88⟩ := decompose_stack Backend.layout5 layoutQ0 rfl h86_0 h87_0 ph73_0 ph85_0
  have t88' := rebase_topRows h88_0 ph74_0 t88
  have b88' := rebase_bottomRows h88_0 ph86_0 b88
  obtain ⟨-, s81, s81'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_80_0
  obtain ⟨-, s89, s89'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_88_0
  obtain ⟨q32a, q32b⟩ := stack2_mul_rows ph32_0 pw_35_0
  obtain ⟨q48a, q48b⟩ := stack2_mul_rows ph48_0 pw_67_0
  obtain ⟨q32a', q32b'⟩ := stack2_mul_rows ph32_0 pw_74_0
  obtain ⟨q48a', q48b'⟩ := stack2_mul_rows ph48_0 pw_86_0
  have k82 : w_82_0 = pw_70_0 := by
    rw [slice_rowOf (by inst_num) h82_0, slice_rowOf (by inst_num) ph70_0]
    show rowOf (w_74_0 * w_80_0) 1 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 0
    rw [s81, pairMul, rowOf_add, q32a, q48a, t80', b80', k55, k71]
    rfl
  have k92 : w_92_0 = pw_91_0 := by
    rw [slice_rowOf (by inst_num) h92_0, slice_rowOf (by inst_num) ph91_0]
    show rowOf (w_74_0 * w_80_0) 2 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 1
    rw [s81', pairMul, rowOf_add, q32b, q48b, t80', b80', k56, k72]
    rfl
  have k90 : w_90_0 = pw_89_0 := by
    rw [slice_rowOf (by inst_num) h90_0, slice_rowOf (by inst_num) ph89_0]
    show rowOf (w_74_0 * w_88_0) 1 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 0
    rw [s89, pairMul, rowOf_add, q32a', q48a', t88', b88', k55, k71]
    rfl
  have k94 : w_94_0 = pw_93_0 := by
    rw [slice_rowOf (by inst_num) h94_0, slice_rowOf (by inst_num) ph93_0]
    show rowOf (w_74_0 * w_88_0) 2 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 1
    rw [s89', pairMul, rowOf_add, q32b', q48b', t88', b88', k56, k72]
    rfl
  rw [k82] at h83_0
  rw [k92] at h93_0
  rw [k90] at h91_0
  rw [k94] at h95_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h83_0 ph71_0, blockModSwitchRuns_unique h91_0 ph90_0,
    blockModSwitchRuns_unique h93_0 ph92_0, blockModSwitchRuns_unique h95_0 ph94_0⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `0` of the smudge PRF (online `771`) are those the
preprocessing computes (`513`). -/
theorem keysCell771 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12} {apay : Fin 5 → ExactMatrix Q6 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : ExactMatrix Q6 N 1 10} {bot0 bot1 : ExactMatrix Q5 N 1 8}
    {outs : Blocks Q5 8}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_773 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q6 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_517 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q6 N 1 12 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_773 at hon
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_6_0, hon⟩ := hon
  obtain ⟨w_7_0, hon⟩ := hon
  obtain ⟨w_7_concat_1, hon⟩ := hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  obtain ⟨w_14_concat_1, hon⟩ := hon
  obtain ⟨w_15_0, hon⟩ := hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_19_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hon
  obtain ⟨w_28_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  obtain ⟨w_30_0, hon⟩ := hon
  obtain ⟨w_30_concat_1, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  obtain ⟨w_34_0, hon⟩ := hon
  obtain ⟨w_35_0, hon⟩ := hon
  extract_lets -merge w_36_0 at hon
  obtain ⟨w_37_0, hon⟩ := hon
  extract_lets -merge w_40_0 at hon
  obtain ⟨w_41_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_43_concat_1, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_46_0, hon⟩ := hon
  obtain ⟨w_48_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_50_0, hon⟩ := hon
  extract_lets -merge w_51_0 at hon
  obtain ⟨w_52_0, hon⟩ := hon
  obtain ⟨w_53_0, hon⟩ := hon
  obtain ⟨w_54_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_58_0, hon⟩ := hon
  extract_lets -merge w_59_0 at hon
  obtain ⟨w_60_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  obtain ⟨w_65_0, hon⟩ := hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_67_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  obtain ⟨w_69_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_773.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_773.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_773.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_773.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_773.constraints_4] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, -, -, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, -, -, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_517 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_4_0 pw_10_0 pw_21_0 pw_24_0 pw_31_0 pw_38_0 pw_40_0 pw_58_0 pw_36_0 pw_39_0 pw_41_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_2_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_5_0, hpre⟩ := hpre
  extract_lets -merge pw_6_0 at hpre
  obtain ⟨pw_7_0, hpre⟩ := hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  extract_lets -merge pw_12_0 pw_13_0 at hpre
  obtain ⟨pw_14_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  obtain ⟨pw_17_0, hpre⟩ := hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  obtain ⟨pw_19_0, hpre⟩ := hpre
  extract_lets -merge pw_20_0 at hpre
  obtain ⟨pw_22_0, hpre⟩ := hpre
  extract_lets -merge pw_23_0 at hpre
  obtain ⟨pw_25_0, hpre⟩ := hpre
  extract_lets -merge pw_26_0 pw_27_0 at hpre
  obtain ⟨pw_28_0, hpre⟩ := hpre
  obtain ⟨pw_29_0, hpre⟩ := hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 at hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  extract_lets -merge pw_35_0 pw_37_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  extract_lets -merge pw_43_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  extract_lets -merge pw_47_0 at hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_49_0, hpre⟩ := hpre
  extract_lets -merge pw_50_0 pw_51_0 at hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  obtain ⟨pw_53_0, hpre⟩ := hpre
  obtain ⟨pw_54_0, hpre⟩ := hpre
  extract_lets -merge pw_63_0 at hpre
  obtain ⟨pw_55_0, hpre⟩ := hpre
  obtain ⟨pw_56_0, hpre⟩ := hpre
  extract_lets -merge pw_57_0 at hpre
  obtain ⟨pw_59_0, hpre⟩ := hpre
  extract_lets -merge pw_60_0 pw_61_0 pw_62_0 pw_64_0 at hpre
  obtain ⟨pw_65_0, hpre⟩ := hpre
  extract_lets -merge pw_66_0 at hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  obtain ⟨pw_68_0, hpre⟩ := hpre
  extract_lets -merge pw_69_0 pw_70_0 at hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  obtain ⟨pw_75_0, hpre⟩ := hpre
  obtain ⟨pw_76_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_517.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_517.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_517.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_517.constraints_3] at hpre
  obtain ⟨-, -, -, ph1_0, -, -, -, ph2_0, ph3_0, ph5_0, -, -, -, ph7_0, -, -, -, ph8_0, ph9_0, ph11_0, -, -, -, -, -, -, ph14_0, -, -, -, -, -, -, ph15_0, ph16_0, -, -, ph17_0, ph18_0, ph19_0, ph22_0, ph25_0, -, -, -, -, -, -, ph28_0, -, -, -, -, -, -, ph29_0, ph30_0, ph32_0, ph33_0, ph34_0, ph42_0, ph46_0, ph48_0, ph49_0, -, -, -, -, -, -, ph52_0, ph53_0, -, -, ph54_0, ph55_0, ph56_0, ph59_0, ph65_0, ph67_0, ph68_0, -, -, -, -, -, -, ph71_0, ph72_0, -, -, -, -, -, -, ph73_0, ph74_0, -, -, -, -, -, -, ph75_0, ph76_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h4_0 h6_0 h11_0 h13_0
  obtain ⟨hk, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_4_0 := by
    rw [hashSample_eq ph1_0, hw4, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e2 : pw_2_0 = w_6_0 := by
    rw [hashSample_eq ph2_0, hw6, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e7 : pw_7_0 = w_11_0 := by
    rw [hashSample_eq ph7_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e8 : pw_8_0 = w_13_0 := by
    rw [hashSample_eq ph8_0, hw13, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The public multiplication by the one encoding.
  obtain ⟨t20, b20⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h18_0 h19_0 ph5_0 ph11_0
  have r20 := rebase_self (by inst_num) h20_0
  obtain ⟨t35, b35⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h33_0 h34_0 ph22_0 ph25_0
  have r35 := rebase_self (by inst_num) h35_0
  obtain ⟨-, s21, s21'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_20_0
  obtain ⟨-, s36, s36'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_35_0
  obtain ⟨p3a, p3b⟩ := stack2_mul_rows ph3_0 pw_5_0
  obtain ⟨p9a, p9b⟩ := stack2_mul_rows ph9_0 pw_11_0
  obtain ⟨p3a', p3b'⟩ := stack2_mul_rows ph3_0 pw_22_0
  obtain ⟨p9a', p9b'⟩ := stack2_mul_rows ph9_0 pw_25_0
  have k28 : w_28_0 = pw_14_0 := by
    rw [slice_rowOf (by inst_num) h28_0, slice_rowOf (by inst_num) ph14_0]
    show rowOf (w_15_0 * w_20_0) 1 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 0
    rw [s21, pairMul, rowOf_add, p3a, p9a, r20, t20, b20, e1, e7]
    rfl
  have k29 : w_29_0 = pw_15_0 := by
    rw [slice_rowOf (by inst_num) h29_0, slice_rowOf (by inst_num) ph15_0]
    show rowOf (w_15_0 * w_20_0) 2 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 1
    rw [s21', pairMul, rowOf_add, p3b, p9b, r20, t20, b20, e2, e8]
    rfl
  have k41 : w_41_0 = pw_28_0 := by
    rw [slice_rowOf (by inst_num) h41_0, slice_rowOf (by inst_num) ph28_0]
    show rowOf (w_15_0 * w_35_0) 1 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 0
    rw [s36, pairMul, rowOf_add, p3a', p9a', r35, t35, b35, e1, e7]
    rfl
  have k42 : w_42_0 = pw_29_0 := by
    rw [slice_rowOf (by inst_num) h42_0, slice_rowOf (by inst_num) ph29_0]
    show rowOf (w_15_0 * w_35_0) 2 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 1
    rw [s36', pairMul, rowOf_add, p3b', p9b', r35, t35, b35, e2, e8]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_46_0 = pw_17_0 := (hashSample_eq h46_0).trans (hashSample_eq ph17_0).symm
  have et1 : w_54_0 = pw_54_0 := (hashSample_eq h54_0).trans (hashSample_eq ph54_0).symm
  have eb0 : w_47_0 = pw_47_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_55_0 = pw_66_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h48_0
  rw [et1, eb1] at h56_0
  obtain ⟨t50, b50⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h48_0 h49_0 ph18_0 ph48_0
  have t50' := rebase_topRows h50_0 ph19_0 t50
  have b50' := rebase_bottomRows h50_0 ph49_0 b50
  obtain ⟨t58, b58⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h56_0 h57_0 ph55_0 ph67_0
  have t58' := rebase_topRows h58_0 ph56_0 t58
  have b58' := rebase_bottomRows h58_0 ph68_0 b58
  obtain ⟨-, s51, s51'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_50_0
  obtain ⟨-, s59, s59'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_58_0
  obtain ⟨q16a, q16b⟩ := stack2_mul_rows ph16_0 pw_19_0
  obtain ⟨q30a, q30b⟩ := stack2_mul_rows ph30_0 pw_49_0
  obtain ⟨q16a', q16b'⟩ := stack2_mul_rows ph16_0 pw_56_0
  obtain ⟨q30a', q30b'⟩ := stack2_mul_rows ph30_0 pw_68_0
  have k52 : w_52_0 = pw_52_0 := by
    rw [slice_rowOf (by inst_num) h52_0, slice_rowOf (by inst_num) ph52_0]
    show rowOf (w_44_0 * w_50_0) 1 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 0
    rw [s51, pairMul, rowOf_add, q16a, q30a, t50', b50', k28, k41]
    rfl
  have k62 : w_62_0 = pw_73_0 := by
    rw [slice_rowOf (by inst_num) h62_0, slice_rowOf (by inst_num) ph73_0]
    show rowOf (w_44_0 * w_50_0) 2 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 1
    rw [s51', pairMul, rowOf_add, q16b, q30b, t50', b50', k29, k42]
    rfl
  have k60 : w_60_0 = pw_71_0 := by
    rw [slice_rowOf (by inst_num) h60_0, slice_rowOf (by inst_num) ph71_0]
    show rowOf (w_44_0 * w_58_0) 1 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 0
    rw [s59, pairMul, rowOf_add, q16a', q30a', t58', b58', k28, k41]
    rfl
  have k64 : w_64_0 = pw_75_0 := by
    rw [slice_rowOf (by inst_num) h64_0, slice_rowOf (by inst_num) ph75_0]
    show rowOf (w_44_0 * w_58_0) 2 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 1
    rw [s59', pairMul, rowOf_add, q16b', q30b', t58', b58', k29, k42]
    rfl
  rw [k52] at h53_0
  rw [k62] at h63_0
  rw [k60] at h61_0
  rw [k64] at h65_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h53_0 ph53_0, blockModSwitchRuns_unique h61_0 ph72_0,
    blockModSwitchRuns_unique h63_0 ph74_0, blockModSwitchRuns_unique h65_0 ph76_0⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `1` of the smudge PRF (online `775`) are those the
preprocessing computes (`517`). -/
theorem keysCell775 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12} {apay : Fin 5 → ExactMatrix Q6 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : Fin 160 → ExactMatrix Q5 N 1 8}
    {bot0 bot1 : ExactMatrix Q0 N 1 6} {outs : Blocks Q0 6}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_777 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q6 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_521 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q6 N 1 12 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_777 at hon
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_5_0, hon⟩ := hon
  extract_lets -merge w_6_0 at hon
  obtain ⟨w_7_0, hon⟩ := hon
  extract_lets -merge w_8_0 at hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  extract_lets -merge w_15_0 at hon
  obtain ⟨w_16_0, hon⟩ := hon
  extract_lets -merge w_17_0 at hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_23_0, hon⟩ := hon
  obtain ⟨w_23_concat_1, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 at hon
  obtain ⟨w_27_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  extract_lets -merge w_30_0 at hon
  obtain ⟨w_31_0, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  extract_lets -merge w_34_0 at hon
  obtain ⟨w_35_0, hon⟩ := hon
  obtain ⟨w_36_0, hon⟩ := hon
  obtain ⟨w_36_concat_1, hon⟩ := hon
  obtain ⟨w_37_0, hon⟩ := hon
  obtain ⟨w_40_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_45_0, hon⟩ := hon
  extract_lets -merge w_46_0 at hon
  obtain ⟨w_47_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_51_0, hon⟩ := hon
  obtain ⟨w_52_0, hon⟩ := hon
  extract_lets -merge w_53_0 w_54_0 at hon
  obtain ⟨w_55_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_57_concat_1, hon⟩ := hon
  obtain ⟨w_59_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  extract_lets -merge w_65_0 at hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  extract_lets -merge w_69_0 w_70_0 at hon
  obtain ⟨w_71_0, hon⟩ := hon
  obtain ⟨w_72_0, hon⟩ := hon
  obtain ⟨w_73_0, hon⟩ := hon
  obtain ⟨w_73_concat_1, hon⟩ := hon
  obtain ⟨w_74_0, hon⟩ := hon
  obtain ⟨w_76_0, hon⟩ := hon
  obtain ⟨w_78_0, hon⟩ := hon
  obtain ⟨w_79_0, hon⟩ := hon
  obtain ⟨w_80_0, hon⟩ := hon
  extract_lets -merge w_81_0 at hon
  obtain ⟨w_82_0, hon⟩ := hon
  obtain ⟨w_83_0, hon⟩ := hon
  obtain ⟨w_84_0, hon⟩ := hon
  obtain ⟨w_86_0, hon⟩ := hon
  obtain ⟨w_87_0, hon⟩ := hon
  obtain ⟨w_88_0, hon⟩ := hon
  extract_lets -merge w_89_0 at hon
  obtain ⟨w_90_0, hon⟩ := hon
  obtain ⟨w_91_0, hon⟩ := hon
  obtain ⟨w_92_0, hon⟩ := hon
  obtain ⟨w_93_0, hon⟩ := hon
  obtain ⟨w_94_0, hon⟩ := hon
  obtain ⟨w_95_0, hon⟩ := hon
  obtain ⟨w_96_0, hon⟩ := hon
  obtain ⟨w_97_0, hon⟩ := hon
  obtain ⟨w_98_0, hon⟩ := hon
  obtain ⟨w_99_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_777.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_777.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_777.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_777.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_777.constraints_4, W_Stage_online_eval.parallel_parallel_generatedRoot_290_777.constraints_5] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, -, -, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, -, -, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_521 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_13_0 pw_25_0 pw_37_0 pw_41_0 pw_49_0 pw_56_0 pw_58_0 pw_76_0 pw_2_0 pw_14_0 pw_54_0 pw_57_0 pw_59_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_4_0, hpre⟩ := hpre
  extract_lets -merge pw_5_0 at hpre
  obtain ⟨pw_6_0, hpre⟩ := hpre
  extract_lets -merge pw_7_0 at hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  extract_lets -merge pw_10_0 at hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  obtain ⟨pw_12_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  extract_lets -merge pw_17_0 at hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  extract_lets -merge pw_19_0 at hpre
  obtain ⟨pw_20_0, hpre⟩ := hpre
  obtain ⟨pw_21_0, hpre⟩ := hpre
  extract_lets -merge pw_22_0 at hpre
  obtain ⟨pw_23_0, hpre⟩ := hpre
  obtain ⟨pw_24_0, hpre⟩ := hpre
  obtain ⟨pw_26_0, hpre⟩ := hpre
  obtain ⟨pw_27_0, hpre⟩ := hpre
  extract_lets -merge pw_28_0 pw_29_0 at hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_31_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  obtain ⟨pw_35_0, hpre⟩ := hpre
  extract_lets -merge pw_36_0 at hpre
  obtain ⟨pw_38_0, hpre⟩ := hpre
  obtain ⟨pw_39_0, hpre⟩ := hpre
  extract_lets -merge pw_40_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  obtain ⟨pw_43_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  obtain ⟨pw_47_0, hpre⟩ := hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_50_0, hpre⟩ := hpre
  extract_lets -merge pw_62_0 at hpre
  obtain ⟨pw_51_0, hpre⟩ := hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  extract_lets -merge pw_53_0 pw_55_0 at hpre
  obtain ⟨pw_60_0, hpre⟩ := hpre
  extract_lets -merge pw_61_0 pw_63_0 at hpre
  obtain ⟨pw_64_0, hpre⟩ := hpre
  extract_lets -merge pw_65_0 at hpre
  obtain ⟨pw_66_0, hpre⟩ := hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  extract_lets -merge pw_68_0 pw_69_0 at hpre
  obtain ⟨pw_70_0, hpre⟩ := hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  extract_lets -merge pw_81_0 at hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  extract_lets -merge pw_75_0 at hpre
  obtain ⟨pw_77_0, hpre⟩ := hpre
  extract_lets -merge pw_78_0 pw_79_0 pw_80_0 pw_82_0 at hpre
  obtain ⟨pw_83_0, hpre⟩ := hpre
  extract_lets -merge pw_84_0 at hpre
  obtain ⟨pw_85_0, hpre⟩ := hpre
  obtain ⟨pw_86_0, hpre⟩ := hpre
  extract_lets -merge pw_87_0 pw_88_0 at hpre
  obtain ⟨pw_89_0, hpre⟩ := hpre
  obtain ⟨pw_90_0, hpre⟩ := hpre
  obtain ⟨pw_91_0, hpre⟩ := hpre
  obtain ⟨pw_92_0, hpre⟩ := hpre
  obtain ⟨pw_93_0, hpre⟩ := hpre
  obtain ⟨pw_94_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_521.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_521.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_521.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_521.constraints_3] at hpre
  obtain ⟨-, -, -, ph1_0, ph3_0, ph4_0, ph6_0, ph8_0, -, -, -, ph9_0, ph11_0, ph12_0, -, ph14_0, ph14_0_1, ph15_0, ph16_0, -, -, -, ph18_0, ph20_0, -, -, -, ph21_0, ph23_0, ph24_0, ph14_0_2, ph14_0_3, ph26_0, ph27_0, -, -, -, -, -, -, ph30_0, -, -, -, -, -, -, ph31_0, ph32_0, -, -, ph33_0, ph34_0, ph35_0, ph14_0_4, ph14_0_5, ph38_0, ph39_0, ph14_0_6, ph14_0_7, ph42_0, ph43_0, -, -, -, -, -, -, ph46_0, -, -, -, -, -, -, ph47_0, ph48_0, ph50_0, ph51_0, ph52_0, -, -, ph60_0, ph64_0, ph66_0, ph67_0, -, -, -, -, -, -, ph70_0, ph71_0, -, -, ph72_0, ph73_0, ph74_0, ph77_0, ph83_0, ph85_0, ph86_0, -, -, -, -, -, -, ph89_0, ph90_0, -, -, -, -, -, -, ph91_0, ph92_0, -, -, -, -, -, -, ph93_0, ph94_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h11_0 h20_0 h29_0 h33_0
  obtain ⟨hk, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_11_0 := by
    rw [hashSample_eq ph1_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e9 : pw_9_0 = w_20_0 := by
    rw [hashSample_eq ph9_0, hw20, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e18 : pw_18_0 = w_29_0 := by
    rw [hashSample_eq ph18_0, hw29, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e21 : pw_21_0 = w_33_0 := by
    rw [hashSample_eq ph21_0, hw33, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The alignment `D(p G)` of the label.
  have g13 : w_13_0 = pw_3_0 := gadgetMatrixRuns_unique h13_0 ph3_0
  rw [g13] at h14_0
  have g14 : w_14_0 = pw_4_0 := centeredRebaseRuns_unique h14_0 ph4_0
  have g16 : w_16_0 = pw_6_0 := by
    apply gadgetDecomposeRuns_unique h16_0
    show gadgetDecomposeRuns _ _ _ (matrixMulScalarLeft _ w_14_0) _
    rw [g14]
    exact ph6_0
  have a18 : w_18_0 = pw_8_0 := by
    apply blockModSwitchRuns_unique h18_0
    show blockModSwitchRuns _ _ (w_11_0 * w_16_0) _
    rw [← e1, g16]
    exact ph8_0
  have a22 : w_22_0 = pw_11_0 := by
    apply blockModSwitchRuns_unique h22_0
    show blockModSwitchRuns _ _ (w_20_0 * w_16_0) _
    rw [← e9, g16]
    exact ph11_0
  have a31 : w_31_0 = pw_20_0 := by
    apply blockModSwitchRuns_unique h31_0
    show blockModSwitchRuns _ _ (w_29_0 * w_16_0) _
    rw [← e18, g16]
    exact ph20_0
  have a35 : w_35_0 = pw_23_0 := by
    apply blockModSwitchRuns_unique h35_0
    show blockModSwitchRuns _ _ (w_33_0 * w_16_0) _
    rw [← e21, g16]
    exact ph23_0
  -- The members of bucket `0`.
  have ho : w_39_0 = pw_14_0 := rfl
  rw [ho] at h40_0 h42_0 h59_0 h61_0
  have m40 : w_40_0 = pw_15_0 := familyGetDynamic_unique h40_0 ph15_0
  have m42 : w_42_0 = pw_26_0 := familyGetDynamic_unique h42_0 ph26_0
  have m59 : w_59_0 = pw_38_0 := familyGetDynamic_unique h59_0 ph38_0
  have m61 : w_61_0 = pw_42_0 := familyGetDynamic_unique h61_0 ph42_0
  rw [m40, m42] at h43_0
  rw [m59, m61] at h62_0
  obtain ⟨t45, b45⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h43_0 h44_0 ph16_0 ph27_0
  have r45 := rebase_self (by inst_num) h45_0
  obtain ⟨t64, b64⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h62_0 h63_0 ph39_0 ph43_0
  have r64 := rebase_self (by inst_num) h64_0
  obtain ⟨-, s46, s46'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_45_0
  obtain ⟨-, s65, s65'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_64_0
  obtain ⟨p12a, p12b⟩ := stack2_mul_rows ph12_0 pw_16_0
  obtain ⟨p24a, p24b⟩ := stack2_mul_rows ph24_0 pw_27_0
  obtain ⟨p12a', p12b'⟩ := stack2_mul_rows ph12_0 pw_39_0
  obtain ⟨p24a', p24b'⟩ := stack2_mul_rows ph24_0 pw_43_0
  have k55 : w_55_0 = pw_30_0 := by
    rw [slice_rowOf (by inst_num) h55_0, slice_rowOf (by inst_num) ph30_0]
    show rowOf (w_37_0 * w_45_0) 1 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 0
    rw [s46, pairMul, rowOf_add, p12a, p24a, r45, t45, b45, a18, a31]
    rfl
  have k56 : w_56_0 = pw_31_0 := by
    rw [slice_rowOf (by inst_num) h56_0, slice_rowOf (by inst_num) ph31_0]
    show rowOf (w_37_0 * w_45_0) 2 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 1
    rw [s46', pairMul, rowOf_add, p12b, p24b, r45, t45, b45, a22, a35]
    rfl
  have k71 : w_71_0 = pw_46_0 := by
    rw [slice_rowOf (by inst_num) h71_0, slice_rowOf (by inst_num) ph46_0]
    show rowOf (w_37_0 * w_64_0) 1 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 0
    rw [s65, pairMul, rowOf_add, p12a', p24a', r64, t64, b64, a18, a31]
    rfl
  have k72 : w_72_0 = pw_47_0 := by
    rw [slice_rowOf (by inst_num) h72_0, slice_rowOf (by inst_num) ph47_0]
    show rowOf (w_37_0 * w_64_0) 2 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 1
    rw [s65', pairMul, rowOf_add, p12b', p24b', r64, t64, b64, a22, a35]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_76_0 = pw_33_0 := (hashSample_eq h76_0).trans (hashSample_eq ph33_0).symm
  have et1 : w_84_0 = pw_72_0 := (hashSample_eq h84_0).trans (hashSample_eq ph72_0).symm
  have eb0 : w_77_0 = pw_65_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_85_0 = pw_84_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h78_0
  rw [et1, eb1] at h86_0
  obtain ⟨t80, b80⟩ := decompose_stack Backend.layout5 layoutQ0 rfl h78_0 h79_0 ph34_0 ph66_0
  have t80' := rebase_topRows h80_0 ph35_0 t80
  have b80' := rebase_bottomRows h80_0 ph67_0 b80
  obtain ⟨t88, b88⟩ := decompose_stack Backend.layout5 layoutQ0 rfl h86_0 h87_0 ph73_0 ph85_0
  have t88' := rebase_topRows h88_0 ph74_0 t88
  have b88' := rebase_bottomRows h88_0 ph86_0 b88
  obtain ⟨-, s81, s81'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_80_0
  obtain ⟨-, s89, s89'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_88_0
  obtain ⟨q32a, q32b⟩ := stack2_mul_rows ph32_0 pw_35_0
  obtain ⟨q48a, q48b⟩ := stack2_mul_rows ph48_0 pw_67_0
  obtain ⟨q32a', q32b'⟩ := stack2_mul_rows ph32_0 pw_74_0
  obtain ⟨q48a', q48b'⟩ := stack2_mul_rows ph48_0 pw_86_0
  have k82 : w_82_0 = pw_70_0 := by
    rw [slice_rowOf (by inst_num) h82_0, slice_rowOf (by inst_num) ph70_0]
    show rowOf (w_74_0 * w_80_0) 1 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 0
    rw [s81, pairMul, rowOf_add, q32a, q48a, t80', b80', k55, k71]
    rfl
  have k92 : w_92_0 = pw_91_0 := by
    rw [slice_rowOf (by inst_num) h92_0, slice_rowOf (by inst_num) ph91_0]
    show rowOf (w_74_0 * w_80_0) 2 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 1
    rw [s81', pairMul, rowOf_add, q32b, q48b, t80', b80', k56, k72]
    rfl
  have k90 : w_90_0 = pw_89_0 := by
    rw [slice_rowOf (by inst_num) h90_0, slice_rowOf (by inst_num) ph89_0]
    show rowOf (w_74_0 * w_88_0) 1 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 0
    rw [s89, pairMul, rowOf_add, q32a', q48a', t88', b88', k55, k71]
    rfl
  have k94 : w_94_0 = pw_93_0 := by
    rw [slice_rowOf (by inst_num) h94_0, slice_rowOf (by inst_num) ph93_0]
    show rowOf (w_74_0 * w_88_0) 2 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 1
    rw [s89', pairMul, rowOf_add, q32b', q48b', t88', b88', k56, k72]
    rfl
  rw [k82] at h83_0
  rw [k92] at h93_0
  rw [k90] at h91_0
  rw [k94] at h95_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h83_0 ph71_0, blockModSwitchRuns_unique h91_0 ph90_0,
    blockModSwitchRuns_unique h93_0 ph92_0, blockModSwitchRuns_unique h95_0 ph94_0⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `0` of the smudge PRF (online `825`) are those the
preprocessing computes (`557`). -/
theorem keysCell825 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12} {apay : Fin 5 → ExactMatrix Q6 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : ExactMatrix Q6 N 1 10} {bot0 bot1 : ExactMatrix Q5 N 1 8}
    {outs : Blocks Q5 8}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_827 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q6 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_561 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q6 N 1 12 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_827 at hon
  extract_lets -merge _params _backend w_0_0 w_3_0 w_5_0 w_8_0 w_10_0 w_12_0 w_16_0 w_17_0 w_23_0 w_24_0 w_31_0 w_32_0 w_38_0 w_45_0 w_47_0 w_55_0 w_1_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_6_0, hon⟩ := hon
  obtain ⟨w_7_0, hon⟩ := hon
  obtain ⟨w_7_concat_1, hon⟩ := hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  obtain ⟨w_14_concat_1, hon⟩ := hon
  obtain ⟨w_15_0, hon⟩ := hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_19_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 w_27_0 w_39_0 at hon
  obtain ⟨w_28_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  obtain ⟨w_30_0, hon⟩ := hon
  obtain ⟨w_30_concat_1, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  obtain ⟨w_34_0, hon⟩ := hon
  obtain ⟨w_35_0, hon⟩ := hon
  extract_lets -merge w_36_0 at hon
  obtain ⟨w_37_0, hon⟩ := hon
  extract_lets -merge w_40_0 at hon
  obtain ⟨w_41_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_43_concat_1, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_46_0, hon⟩ := hon
  obtain ⟨w_48_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_50_0, hon⟩ := hon
  extract_lets -merge w_51_0 at hon
  obtain ⟨w_52_0, hon⟩ := hon
  obtain ⟨w_53_0, hon⟩ := hon
  obtain ⟨w_54_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_58_0, hon⟩ := hon
  extract_lets -merge w_59_0 at hon
  obtain ⟨w_60_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  obtain ⟨w_65_0, hon⟩ := hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_67_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  obtain ⟨w_69_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_827.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_827.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_827.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_827.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_827.constraints_4] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h1_0_2, h1_0_3, h4_0, h1_0_4, h1_0_5, h6_0, h4_0_1, h7_0, h1_0_6, h1_0_7, h9_0, h1_0_8, h1_0_9, h11_0, h1_0_10, h1_0_11, h13_0, h11_0_1, h14_0, h15_0, h18_0, h19_0, h20_0, -, -, -, -, -, -, h22_0, h1_0_12, h1_0_13, h25_0, -, -, -, -, -, -, h28_0, -, -, -, -, -, -, h29_0, h28_0_1, h30_0, h33_0, h34_0, h35_0, -, -, -, -, -, -, h37_0, -, -, -, -, -, -, h41_0, -, -, -, -, -, -, h42_0, h41_0_1, h43_0, h44_0, -, -, h46_0, h48_0, h49_0, h50_0, -, -, -, -, -, -, h52_0, h53_0, -, -, h54_0, h56_0, h57_0, h58_0, -, -, -, -, -, -, h60_0, h61_0, -, -, -, -, -, -, h62_0, h63_0, -, -, -, -, -, -, h64_0, h65_0, -, -, -, -, -, -, h66_0, h67_0, -, -, -, -, -, -, h68_0, h69_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_561 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_4_0 pw_10_0 pw_21_0 pw_24_0 pw_31_0 pw_38_0 pw_40_0 pw_58_0 pw_36_0 pw_39_0 pw_41_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_2_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_5_0, hpre⟩ := hpre
  extract_lets -merge pw_6_0 at hpre
  obtain ⟨pw_7_0, hpre⟩ := hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  extract_lets -merge pw_12_0 pw_13_0 at hpre
  obtain ⟨pw_14_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  obtain ⟨pw_17_0, hpre⟩ := hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  obtain ⟨pw_19_0, hpre⟩ := hpre
  extract_lets -merge pw_20_0 at hpre
  obtain ⟨pw_22_0, hpre⟩ := hpre
  extract_lets -merge pw_23_0 at hpre
  obtain ⟨pw_25_0, hpre⟩ := hpre
  extract_lets -merge pw_26_0 pw_27_0 at hpre
  obtain ⟨pw_28_0, hpre⟩ := hpre
  obtain ⟨pw_29_0, hpre⟩ := hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 at hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  extract_lets -merge pw_35_0 pw_37_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  extract_lets -merge pw_43_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  extract_lets -merge pw_47_0 at hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_49_0, hpre⟩ := hpre
  extract_lets -merge pw_50_0 pw_51_0 at hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  obtain ⟨pw_53_0, hpre⟩ := hpre
  obtain ⟨pw_54_0, hpre⟩ := hpre
  extract_lets -merge pw_63_0 at hpre
  obtain ⟨pw_55_0, hpre⟩ := hpre
  obtain ⟨pw_56_0, hpre⟩ := hpre
  extract_lets -merge pw_57_0 at hpre
  obtain ⟨pw_59_0, hpre⟩ := hpre
  extract_lets -merge pw_60_0 pw_61_0 pw_62_0 pw_64_0 at hpre
  obtain ⟨pw_65_0, hpre⟩ := hpre
  extract_lets -merge pw_66_0 at hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  obtain ⟨pw_68_0, hpre⟩ := hpre
  extract_lets -merge pw_69_0 pw_70_0 at hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  obtain ⟨pw_75_0, hpre⟩ := hpre
  obtain ⟨pw_76_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_561.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_561.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_561.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_561.constraints_3] at hpre
  obtain ⟨-, -, -, ph1_0, -, -, -, ph2_0, ph3_0, ph5_0, -, -, -, ph7_0, -, -, -, ph8_0, ph9_0, ph11_0, -, -, -, -, -, -, ph14_0, -, -, -, -, -, -, ph15_0, ph16_0, -, -, ph17_0, ph18_0, ph19_0, ph22_0, ph25_0, -, -, -, -, -, -, ph28_0, -, -, -, -, -, -, ph29_0, ph30_0, ph32_0, ph33_0, ph34_0, ph42_0, ph46_0, ph48_0, ph49_0, -, -, -, -, -, -, ph52_0, ph53_0, -, -, ph54_0, ph55_0, ph56_0, ph59_0, ph65_0, ph67_0, ph68_0, -, -, -, -, -, -, ph71_0, ph72_0, -, -, -, -, -, -, ph73_0, ph74_0, -, -, -, -, -, -, ph75_0, ph76_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h4_0 h6_0 h11_0 h13_0
  obtain ⟨hk, hw4⟩ := familyGet_eq h4_0
  obtain ⟨_, hw6⟩ := familyGet_eq h6_0
  obtain ⟨_, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw13⟩ := familyGet_eq h13_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_4_0 := by
    rw [hashSample_eq ph1_0, hw4, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e2 : pw_2_0 = w_6_0 := by
    rw [hashSample_eq ph2_0, hw6, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e7 : pw_7_0 = w_11_0 := by
    rw [hashSample_eq ph7_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e8 : pw_8_0 = w_13_0 := by
    rw [hashSample_eq ph8_0, hw13, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The public multiplication by the one encoding.
  obtain ⟨t20, b20⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h18_0 h19_0 ph5_0 ph11_0
  have r20 := rebase_self (by inst_num) h20_0
  obtain ⟨t35, b35⟩ := decompose_stack Backend.layout7 layoutQ6 rfl h33_0 h34_0 ph22_0 ph25_0
  have r35 := rebase_self (by inst_num) h35_0
  obtain ⟨-, s21, s21'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_20_0
  obtain ⟨-, s36, s36'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_35_0
  obtain ⟨p3a, p3b⟩ := stack2_mul_rows ph3_0 pw_5_0
  obtain ⟨p9a, p9b⟩ := stack2_mul_rows ph9_0 pw_11_0
  obtain ⟨p3a', p3b'⟩ := stack2_mul_rows ph3_0 pw_22_0
  obtain ⟨p9a', p9b'⟩ := stack2_mul_rows ph9_0 pw_25_0
  have k28 : w_28_0 = pw_14_0 := by
    rw [slice_rowOf (by inst_num) h28_0, slice_rowOf (by inst_num) ph14_0]
    show rowOf (w_15_0 * w_20_0) 1 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 0
    rw [s21, pairMul, rowOf_add, p3a, p9a, r20, t20, b20, e1, e7]
    rfl
  have k29 : w_29_0 = pw_15_0 := by
    rw [slice_rowOf (by inst_num) h29_0, slice_rowOf (by inst_num) ph15_0]
    show rowOf (w_15_0 * w_20_0) 2 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 1
    rw [s21', pairMul, rowOf_add, p3b, p9b, r20, t20, b20, e2, e8]
    rfl
  have k41 : w_41_0 = pw_28_0 := by
    rw [slice_rowOf (by inst_num) h41_0, slice_rowOf (by inst_num) ph28_0]
    show rowOf (w_15_0 * w_35_0) 1 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 0
    rw [s36, pairMul, rowOf_add, p3a', p9a', r35, t35, b35, e1, e7]
    rfl
  have k42 : w_42_0 = pw_29_0 := by
    rw [slice_rowOf (by inst_num) h42_0, slice_rowOf (by inst_num) ph29_0]
    show rowOf (w_15_0 * w_35_0) 2 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 1
    rw [s36', pairMul, rowOf_add, p3b', p9b', r35, t35, b35, e2, e8]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_46_0 = pw_17_0 := (hashSample_eq h46_0).trans (hashSample_eq ph17_0).symm
  have et1 : w_54_0 = pw_54_0 := (hashSample_eq h54_0).trans (hashSample_eq ph54_0).symm
  have eb0 : w_47_0 = pw_47_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_55_0 = pw_66_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h48_0
  rw [et1, eb1] at h56_0
  obtain ⟨t50, b50⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h48_0 h49_0 ph18_0 ph48_0
  have t50' := rebase_topRows h50_0 ph19_0 t50
  have b50' := rebase_bottomRows h50_0 ph49_0 b50
  obtain ⟨t58, b58⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h56_0 h57_0 ph55_0 ph67_0
  have t58' := rebase_topRows h58_0 ph56_0 t58
  have b58' := rebase_bottomRows h58_0 ph68_0 b58
  obtain ⟨-, s51, s51'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_50_0
  obtain ⟨-, s59, s59'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_58_0
  obtain ⟨q16a, q16b⟩ := stack2_mul_rows ph16_0 pw_19_0
  obtain ⟨q30a, q30b⟩ := stack2_mul_rows ph30_0 pw_49_0
  obtain ⟨q16a', q16b'⟩ := stack2_mul_rows ph16_0 pw_56_0
  obtain ⟨q30a', q30b'⟩ := stack2_mul_rows ph30_0 pw_68_0
  have k52 : w_52_0 = pw_52_0 := by
    rw [slice_rowOf (by inst_num) h52_0, slice_rowOf (by inst_num) ph52_0]
    show rowOf (w_44_0 * w_50_0) 1 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 0
    rw [s51, pairMul, rowOf_add, q16a, q30a, t50', b50', k28, k41]
    rfl
  have k62 : w_62_0 = pw_73_0 := by
    rw [slice_rowOf (by inst_num) h62_0, slice_rowOf (by inst_num) ph73_0]
    show rowOf (w_44_0 * w_50_0) 2 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 1
    rw [s51', pairMul, rowOf_add, q16b, q30b, t50', b50', k29, k42]
    rfl
  have k60 : w_60_0 = pw_71_0 := by
    rw [slice_rowOf (by inst_num) h60_0, slice_rowOf (by inst_num) ph71_0]
    show rowOf (w_44_0 * w_58_0) 1 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 0
    rw [s59, pairMul, rowOf_add, q16a', q30a', t58', b58', k28, k41]
    rfl
  have k64 : w_64_0 = pw_75_0 := by
    rw [slice_rowOf (by inst_num) h64_0, slice_rowOf (by inst_num) ph75_0]
    show rowOf (w_44_0 * w_58_0) 2 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 1
    rw [s59', pairMul, rowOf_add, q16b', q30b', t58', b58', k29, k42]
    rfl
  rw [k52] at h53_0
  rw [k62] at h63_0
  rw [k60] at h61_0
  rw [k64] at h65_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h53_0 ph53_0, blockModSwitchRuns_unique h61_0 ph72_0,
    blockModSwitchRuns_unique h63_0 ph74_0, blockModSwitchRuns_unique h65_0 ph76_0⟩

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `1` of the smudge PRF (online `829`) are those the
preprocessing computes (`561`). -/
theorem keysCell829 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params} {params' : W_Stage_preprocess.Params}
    {key : ByteArray} {i : Nat}
    {ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix Q6 N 1 12} {apay : Fin 5 → ExactMatrix Q6 N 1 1}
    {As00 As10 As01 As11 cs0 cs1 : Fin 160 → ExactMatrix Q5 N 1 8}
    {bot0 bot1 : ExactMatrix Q0 N 1 6} {outs : Blocks Q0 6}
    (hon : W_Stage_online_eval.parallel_parallel_generatedRoot_290_831 Backend.backend model params i0 i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {sec one : ExactMatrix Q6 N 1 1} {s e : Int} {pouts : _}
    (hpre : W_Stage_preprocess.parallel_parallel_generatedRoot_407_565 Backend.backend model params' i0 i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample Q6 N 1 12 key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
  unfold W_Stage_online_eval.parallel_parallel_generatedRoot_290_831 at hon
  extract_lets -merge _params _backend w_0_0 w_10_0 w_19_0 w_24_0 w_28_0 w_32_0 w_38_0 w_41_0 w_48_0 w_50_0 w_58_0 w_60_0 w_67_0 w_75_0 w_77_0 w_85_0 w_1_0 w_3_0 w_12_0 w_39_0 at hon
  obtain ⟨w_2_0, hon⟩ := hon
  obtain ⟨w_4_0, hon⟩ := hon
  obtain ⟨w_5_0, hon⟩ := hon
  extract_lets -merge w_6_0 at hon
  obtain ⟨w_7_0, hon⟩ := hon
  extract_lets -merge w_8_0 at hon
  obtain ⟨w_9_0, hon⟩ := hon
  obtain ⟨w_11_0, hon⟩ := hon
  obtain ⟨w_13_0, hon⟩ := hon
  obtain ⟨w_14_0, hon⟩ := hon
  extract_lets -merge w_15_0 at hon
  obtain ⟨w_16_0, hon⟩ := hon
  extract_lets -merge w_17_0 at hon
  obtain ⟨w_18_0, hon⟩ := hon
  obtain ⟨w_20_0, hon⟩ := hon
  extract_lets -merge w_21_0 at hon
  obtain ⟨w_22_0, hon⟩ := hon
  obtain ⟨w_23_0, hon⟩ := hon
  obtain ⟨w_23_concat_1, hon⟩ := hon
  obtain ⟨w_25_0, hon⟩ := hon
  extract_lets -merge w_26_0 at hon
  obtain ⟨w_27_0, hon⟩ := hon
  obtain ⟨w_29_0, hon⟩ := hon
  extract_lets -merge w_30_0 at hon
  obtain ⟨w_31_0, hon⟩ := hon
  obtain ⟨w_33_0, hon⟩ := hon
  extract_lets -merge w_34_0 at hon
  obtain ⟨w_35_0, hon⟩ := hon
  obtain ⟨w_36_0, hon⟩ := hon
  obtain ⟨w_36_concat_1, hon⟩ := hon
  obtain ⟨w_37_0, hon⟩ := hon
  obtain ⟨w_40_0, hon⟩ := hon
  obtain ⟨w_42_0, hon⟩ := hon
  obtain ⟨w_43_0, hon⟩ := hon
  obtain ⟨w_44_0, hon⟩ := hon
  obtain ⟨w_45_0, hon⟩ := hon
  extract_lets -merge w_46_0 at hon
  obtain ⟨w_47_0, hon⟩ := hon
  obtain ⟨w_49_0, hon⟩ := hon
  obtain ⟨w_51_0, hon⟩ := hon
  obtain ⟨w_52_0, hon⟩ := hon
  extract_lets -merge w_53_0 w_54_0 at hon
  obtain ⟨w_55_0, hon⟩ := hon
  obtain ⟨w_56_0, hon⟩ := hon
  obtain ⟨w_57_0, hon⟩ := hon
  obtain ⟨w_57_concat_1, hon⟩ := hon
  obtain ⟨w_59_0, hon⟩ := hon
  obtain ⟨w_61_0, hon⟩ := hon
  obtain ⟨w_62_0, hon⟩ := hon
  obtain ⟨w_63_0, hon⟩ := hon
  obtain ⟨w_64_0, hon⟩ := hon
  extract_lets -merge w_65_0 at hon
  obtain ⟨w_66_0, hon⟩ := hon
  obtain ⟨w_68_0, hon⟩ := hon
  extract_lets -merge w_69_0 w_70_0 at hon
  obtain ⟨w_71_0, hon⟩ := hon
  obtain ⟨w_72_0, hon⟩ := hon
  obtain ⟨w_73_0, hon⟩ := hon
  obtain ⟨w_73_concat_1, hon⟩ := hon
  obtain ⟨w_74_0, hon⟩ := hon
  obtain ⟨w_76_0, hon⟩ := hon
  obtain ⟨w_78_0, hon⟩ := hon
  obtain ⟨w_79_0, hon⟩ := hon
  obtain ⟨w_80_0, hon⟩ := hon
  extract_lets -merge w_81_0 at hon
  obtain ⟨w_82_0, hon⟩ := hon
  obtain ⟨w_83_0, hon⟩ := hon
  obtain ⟨w_84_0, hon⟩ := hon
  obtain ⟨w_86_0, hon⟩ := hon
  obtain ⟨w_87_0, hon⟩ := hon
  obtain ⟨w_88_0, hon⟩ := hon
  extract_lets -merge w_89_0 at hon
  obtain ⟨w_90_0, hon⟩ := hon
  obtain ⟨w_91_0, hon⟩ := hon
  obtain ⟨w_92_0, hon⟩ := hon
  obtain ⟨w_93_0, hon⟩ := hon
  obtain ⟨w_94_0, hon⟩ := hon
  obtain ⟨w_95_0, hon⟩ := hon
  obtain ⟨w_96_0, hon⟩ := hon
  obtain ⟨w_97_0, hon⟩ := hon
  obtain ⟨w_98_0, hon⟩ := hon
  obtain ⟨w_99_0, hon⟩ := hon
  simp only [W_Stage_online_eval.parallel_parallel_generatedRoot_290_831.constraints_0, W_Stage_online_eval.parallel_parallel_generatedRoot_290_831.constraints_1, W_Stage_online_eval.parallel_parallel_generatedRoot_290_831.constraints_2, W_Stage_online_eval.parallel_parallel_generatedRoot_290_831.constraints_3, W_Stage_online_eval.parallel_parallel_generatedRoot_290_831.constraints_4, W_Stage_online_eval.parallel_parallel_generatedRoot_290_831.constraints_5] at hon
  obtain ⟨-, -, -, h1_0, h1_0_1, h2_0, h4_0, h5_0, h7_0, h9_0, h1_0_2, h1_0_3, h11_0, h13_0, h14_0, h16_0, h18_0, h1_0_4, h1_0_5, h20_0, h22_0, h18_0_1, h23_0, h1_0_6, h1_0_7, h25_0, h27_0, h1_0_8, h1_0_9, h29_0, h31_0, h1_0_10, h1_0_11, h33_0, h35_0, h31_0_1, h36_0, h37_0, -, h39_0, h39_0_1, h40_0, h39_0_2, h39_0_3, h42_0, h43_0, h44_0, h45_0, -, -, -, -, -, -, h47_0, h39_0_4, h39_0_5, h49_0, h1_0_12, h1_0_13, h51_0, h52_0, -, -, -, -, -, -, h55_0, -, -, -, -, -, -, h56_0, h55_0_1, h57_0, h39_0_6, h39_0_7, h59_0, h39_0_8, h39_0_9, h61_0, h62_0, h63_0, h64_0, -, -, -, -, -, -, h66_0, h39_0_10, h39_0_11, h68_0, -, -, -, -, -, -, h71_0, -, -, -, -, -, -, h72_0, h71_0_1, h73_0, h74_0, -, -, h76_0, h78_0, h79_0, h80_0, -, -, -, -, -, -, h82_0, h83_0, -, -, h84_0, h86_0, h87_0, h88_0, -, -, -, -, -, -, h90_0, h91_0, -, -, -, -, -, -, h92_0, h93_0, -, -, -, -, -, -, h94_0, h95_0, -, -, -, -, -, -, h96_0, h97_0, -, -, -, -, -, -, h98_0, h99_0, hout⟩ := hon
  unfold W_Stage_preprocess.parallel_parallel_generatedRoot_407_565 at hpre
  extract_lets -merge _params _backend pw_0_0 pw_13_0 pw_25_0 pw_37_0 pw_41_0 pw_49_0 pw_56_0 pw_58_0 pw_76_0 pw_2_0 pw_14_0 pw_54_0 pw_57_0 pw_59_0 at hpre
  obtain ⟨pw_1_0, hpre⟩ := hpre
  obtain ⟨pw_3_0, hpre⟩ := hpre
  obtain ⟨pw_4_0, hpre⟩ := hpre
  extract_lets -merge pw_5_0 at hpre
  obtain ⟨pw_6_0, hpre⟩ := hpre
  extract_lets -merge pw_7_0 at hpre
  obtain ⟨pw_8_0, hpre⟩ := hpre
  obtain ⟨pw_9_0, hpre⟩ := hpre
  extract_lets -merge pw_10_0 at hpre
  obtain ⟨pw_11_0, hpre⟩ := hpre
  obtain ⟨pw_12_0, hpre⟩ := hpre
  obtain ⟨pw_15_0, hpre⟩ := hpre
  obtain ⟨pw_16_0, hpre⟩ := hpre
  extract_lets -merge pw_17_0 at hpre
  obtain ⟨pw_18_0, hpre⟩ := hpre
  extract_lets -merge pw_19_0 at hpre
  obtain ⟨pw_20_0, hpre⟩ := hpre
  obtain ⟨pw_21_0, hpre⟩ := hpre
  extract_lets -merge pw_22_0 at hpre
  obtain ⟨pw_23_0, hpre⟩ := hpre
  obtain ⟨pw_24_0, hpre⟩ := hpre
  obtain ⟨pw_26_0, hpre⟩ := hpre
  obtain ⟨pw_27_0, hpre⟩ := hpre
  extract_lets -merge pw_28_0 pw_29_0 at hpre
  obtain ⟨pw_30_0, hpre⟩ := hpre
  obtain ⟨pw_31_0, hpre⟩ := hpre
  obtain ⟨pw_32_0, hpre⟩ := hpre
  obtain ⟨pw_33_0, hpre⟩ := hpre
  obtain ⟨pw_34_0, hpre⟩ := hpre
  obtain ⟨pw_35_0, hpre⟩ := hpre
  extract_lets -merge pw_36_0 at hpre
  obtain ⟨pw_38_0, hpre⟩ := hpre
  obtain ⟨pw_39_0, hpre⟩ := hpre
  extract_lets -merge pw_40_0 at hpre
  obtain ⟨pw_42_0, hpre⟩ := hpre
  obtain ⟨pw_43_0, hpre⟩ := hpre
  extract_lets -merge pw_44_0 pw_45_0 at hpre
  obtain ⟨pw_46_0, hpre⟩ := hpre
  obtain ⟨pw_47_0, hpre⟩ := hpre
  obtain ⟨pw_48_0, hpre⟩ := hpre
  obtain ⟨pw_50_0, hpre⟩ := hpre
  extract_lets -merge pw_62_0 at hpre
  obtain ⟨pw_51_0, hpre⟩ := hpre
  obtain ⟨pw_52_0, hpre⟩ := hpre
  extract_lets -merge pw_53_0 pw_55_0 at hpre
  obtain ⟨pw_60_0, hpre⟩ := hpre
  extract_lets -merge pw_61_0 pw_63_0 at hpre
  obtain ⟨pw_64_0, hpre⟩ := hpre
  extract_lets -merge pw_65_0 at hpre
  obtain ⟨pw_66_0, hpre⟩ := hpre
  obtain ⟨pw_67_0, hpre⟩ := hpre
  extract_lets -merge pw_68_0 pw_69_0 at hpre
  obtain ⟨pw_70_0, hpre⟩ := hpre
  obtain ⟨pw_71_0, hpre⟩ := hpre
  obtain ⟨pw_72_0, hpre⟩ := hpre
  extract_lets -merge pw_81_0 at hpre
  obtain ⟨pw_73_0, hpre⟩ := hpre
  obtain ⟨pw_74_0, hpre⟩ := hpre
  extract_lets -merge pw_75_0 at hpre
  obtain ⟨pw_77_0, hpre⟩ := hpre
  extract_lets -merge pw_78_0 pw_79_0 pw_80_0 pw_82_0 at hpre
  obtain ⟨pw_83_0, hpre⟩ := hpre
  extract_lets -merge pw_84_0 at hpre
  obtain ⟨pw_85_0, hpre⟩ := hpre
  obtain ⟨pw_86_0, hpre⟩ := hpre
  extract_lets -merge pw_87_0 pw_88_0 at hpre
  obtain ⟨pw_89_0, hpre⟩ := hpre
  obtain ⟨pw_90_0, hpre⟩ := hpre
  obtain ⟨pw_91_0, hpre⟩ := hpre
  obtain ⟨pw_92_0, hpre⟩ := hpre
  obtain ⟨pw_93_0, hpre⟩ := hpre
  obtain ⟨pw_94_0, hpre⟩ := hpre
  simp only [W_Stage_preprocess.parallel_parallel_generatedRoot_407_565.constraints_0, W_Stage_preprocess.parallel_parallel_generatedRoot_407_565.constraints_1, W_Stage_preprocess.parallel_parallel_generatedRoot_407_565.constraints_2, W_Stage_preprocess.parallel_parallel_generatedRoot_407_565.constraints_3] at hpre
  obtain ⟨-, -, -, ph1_0, ph3_0, ph4_0, ph6_0, ph8_0, -, -, -, ph9_0, ph11_0, ph12_0, -, ph14_0, ph14_0_1, ph15_0, ph16_0, -, -, -, ph18_0, ph20_0, -, -, -, ph21_0, ph23_0, ph24_0, ph14_0_2, ph14_0_3, ph26_0, ph27_0, -, -, -, -, -, -, ph30_0, -, -, -, -, -, -, ph31_0, ph32_0, -, -, ph33_0, ph34_0, ph35_0, ph14_0_4, ph14_0_5, ph38_0, ph39_0, ph14_0_6, ph14_0_7, ph42_0, ph43_0, -, -, -, -, -, -, ph46_0, -, -, -, -, -, -, ph47_0, ph48_0, ph50_0, ph51_0, ph52_0, -, -, ph60_0, ph64_0, ph66_0, ph67_0, -, -, -, -, -, -, ph70_0, ph71_0, -, -, ph72_0, ph73_0, ph74_0, ph77_0, ph83_0, ph85_0, ph86_0, -, -, -, -, -, -, ph89_0, ph90_0, -, -, -, -, -, -, ph91_0, ph92_0, -, -, -, -, -, -, ph93_0, ph94_0, phout⟩ := hpre
  -- The label members.
  have hc : w_1_0 = ((i % 5 : Nat) : Int) := select_slot i
  rw [hc] at h11_0 h20_0 h29_0 h33_0
  obtain ⟨hk, hw11⟩ := familyGet_eq h11_0
  obtain ⟨_, hw20⟩ := familyGet_eq h20_0
  obtain ⟨_, hw29⟩ := familyGet_eq h29_0
  obtain ⟨_, hw33⟩ := familyGet_eq h33_0
  have hsel := select_slot i
  have e1 : pw_1_0 = w_11_0 := by
    rw [hashSample_eq ph1_0, hw11, hsel]; exact (haA ⟨i % 5, hk⟩ 0).symm
  have e9 : pw_9_0 = w_20_0 := by
    rw [hashSample_eq ph9_0, hw20, hsel]; exact (haA ⟨i % 5, hk⟩ 2).symm
  have e18 : pw_18_0 = w_29_0 := by
    rw [hashSample_eq ph18_0, hw29, hsel]; exact (haA ⟨i % 5, hk⟩ 1).symm
  have e21 : pw_21_0 = w_33_0 := by
    rw [hashSample_eq ph21_0, hw33, hsel]; exact (haA ⟨i % 5, hk⟩ 3).symm
  -- The alignment `D(p G)` of the label.
  have g13 : w_13_0 = pw_3_0 := gadgetMatrixRuns_unique h13_0 ph3_0
  rw [g13] at h14_0
  have g14 : w_14_0 = pw_4_0 := centeredRebaseRuns_unique h14_0 ph4_0
  have g16 : w_16_0 = pw_6_0 := by
    apply gadgetDecomposeRuns_unique h16_0
    show gadgetDecomposeRuns _ _ _ (matrixMulScalarLeft _ w_14_0) _
    rw [g14]
    exact ph6_0
  have a18 : w_18_0 = pw_8_0 := by
    apply blockModSwitchRuns_unique h18_0
    show blockModSwitchRuns _ _ (w_11_0 * w_16_0) _
    rw [← e1, g16]
    exact ph8_0
  have a22 : w_22_0 = pw_11_0 := by
    apply blockModSwitchRuns_unique h22_0
    show blockModSwitchRuns _ _ (w_20_0 * w_16_0) _
    rw [← e9, g16]
    exact ph11_0
  have a31 : w_31_0 = pw_20_0 := by
    apply blockModSwitchRuns_unique h31_0
    show blockModSwitchRuns _ _ (w_29_0 * w_16_0) _
    rw [← e18, g16]
    exact ph20_0
  have a35 : w_35_0 = pw_23_0 := by
    apply blockModSwitchRuns_unique h35_0
    show blockModSwitchRuns _ _ (w_33_0 * w_16_0) _
    rw [← e21, g16]
    exact ph23_0
  -- The members of bucket `0`.
  have ho : w_39_0 = pw_14_0 := rfl
  rw [ho] at h40_0 h42_0 h59_0 h61_0
  have m40 : w_40_0 = pw_15_0 := familyGetDynamic_unique h40_0 ph15_0
  have m42 : w_42_0 = pw_26_0 := familyGetDynamic_unique h42_0 ph26_0
  have m59 : w_59_0 = pw_38_0 := familyGetDynamic_unique h59_0 ph38_0
  have m61 : w_61_0 = pw_42_0 := familyGetDynamic_unique h61_0 ph42_0
  rw [m40, m42] at h43_0
  rw [m59, m61] at h62_0
  obtain ⟨t45, b45⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h43_0 h44_0 ph16_0 ph27_0
  have r45 := rebase_self (by inst_num) h45_0
  obtain ⟨t64, b64⟩ := decompose_stack Backend.layout6 layoutQ5 rfl h62_0 h63_0 ph39_0 ph43_0
  have r64 := rebase_self (by inst_num) h64_0
  obtain ⟨-, s46, s46'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_45_0
  obtain ⟨-, s65, s65'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_64_0
  obtain ⟨p12a, p12b⟩ := stack2_mul_rows ph12_0 pw_16_0
  obtain ⟨p24a, p24b⟩ := stack2_mul_rows ph24_0 pw_27_0
  obtain ⟨p12a', p12b'⟩ := stack2_mul_rows ph12_0 pw_39_0
  obtain ⟨p24a', p24b'⟩ := stack2_mul_rows ph24_0 pw_43_0
  have k55 : w_55_0 = pw_30_0 := by
    rw [slice_rowOf (by inst_num) h55_0, slice_rowOf (by inst_num) ph30_0]
    show rowOf (w_37_0 * w_45_0) 1 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 0
    rw [s46, pairMul, rowOf_add, p12a, p24a, r45, t45, b45, a18, a31]
    rfl
  have k56 : w_56_0 = pw_31_0 := by
    rw [slice_rowOf (by inst_num) h56_0, slice_rowOf (by inst_num) ph31_0]
    show rowOf (w_37_0 * w_45_0) 2 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 1
    rw [s46', pairMul, rowOf_add, p12b, p24b, r45, t45, b45, a22, a35]
    rfl
  have k71 : w_71_0 = pw_46_0 := by
    rw [slice_rowOf (by inst_num) h71_0, slice_rowOf (by inst_num) ph46_0]
    show rowOf (w_37_0 * w_64_0) 1 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 0
    rw [s65, pairMul, rowOf_add, p12a', p24a', r64, t64, b64, a18, a31]
    rfl
  have k72 : w_72_0 = pw_47_0 := by
    rw [slice_rowOf (by inst_num) h72_0, slice_rowOf (by inst_num) ph47_0]
    show rowOf (w_37_0 * w_64_0) 2 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 1
    rw [s65', pairMul, rowOf_add, p12b', p24b', r64, t64, b64, a22, a35]
    rfl
  -- The private multiplication by the RHS.
  have et0 : w_76_0 = pw_33_0 := (hashSample_eq h76_0).trans (hashSample_eq ph33_0).symm
  have et1 : w_84_0 = pw_72_0 := (hashSample_eq h84_0).trans (hashSample_eq ph72_0).symm
  have eb0 : w_77_0 = pw_65_0 := by show bot0 = _; rw [hbot0, phout]
  have eb1 : w_85_0 = pw_84_0 := by show bot1 = _; rw [hbot1, phout]
  rw [et0, eb0] at h78_0
  rw [et1, eb1] at h86_0
  obtain ⟨t80, b80⟩ := decompose_stack Backend.layout5 layoutQ0 rfl h78_0 h79_0 ph34_0 ph66_0
  have t80' := rebase_topRows h80_0 ph35_0 t80
  have b80' := rebase_bottomRows h80_0 ph67_0 b80
  obtain ⟨t88, b88⟩ := decompose_stack Backend.layout5 layoutQ0 rfl h86_0 h87_0 ph73_0 ph85_0
  have t88' := rebase_topRows h88_0 ph74_0 t88
  have b88' := rebase_bottomRows h88_0 ph86_0 b88
  obtain ⟨-, s81, s81'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_80_0
  obtain ⟨-, s89, s89'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_88_0
  obtain ⟨q32a, q32b⟩ := stack2_mul_rows ph32_0 pw_35_0
  obtain ⟨q48a, q48b⟩ := stack2_mul_rows ph48_0 pw_67_0
  obtain ⟨q32a', q32b'⟩ := stack2_mul_rows ph32_0 pw_74_0
  obtain ⟨q48a', q48b'⟩ := stack2_mul_rows ph48_0 pw_86_0
  have k82 : w_82_0 = pw_70_0 := by
    rw [slice_rowOf (by inst_num) h82_0, slice_rowOf (by inst_num) ph70_0]
    show rowOf (w_74_0 * w_80_0) 1 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 0
    rw [s81, pairMul, rowOf_add, q32a, q48a, t80', b80', k55, k71]
    rfl
  have k92 : w_92_0 = pw_91_0 := by
    rw [slice_rowOf (by inst_num) h92_0, slice_rowOf (by inst_num) ph91_0]
    show rowOf (w_74_0 * w_80_0) 2 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 1
    rw [s81', pairMul, rowOf_add, q32b, q48b, t80', b80', k56, k72]
    rfl
  have k90 : w_90_0 = pw_89_0 := by
    rw [slice_rowOf (by inst_num) h90_0, slice_rowOf (by inst_num) ph89_0]
    show rowOf (w_74_0 * w_88_0) 1 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 0
    rw [s89, pairMul, rowOf_add, q32a', q48a', t88', b88', k55, k71]
    rfl
  have k94 : w_94_0 = pw_93_0 := by
    rw [slice_rowOf (by inst_num) h94_0, slice_rowOf (by inst_num) ph93_0]
    show rowOf (w_74_0 * w_88_0) 2 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 1
    rw [s89', pairMul, rowOf_add, q32b', q48b', t88', b88', k56, k72]
    rfl
  rw [k82] at h83_0
  rw [k92] at h93_0
  rw [k90] at h91_0
  rw [k94] at h95_0
  rw [hout, phout]
  exact ⟨blockModSwitchRuns_unique h83_0 ph71_0, blockModSwitchRuns_unique h91_0 ph90_0,
    blockModSwitchRuns_unique h93_0 ph92_0, blockModSwitchRuns_unique h95_0 ph94_0⟩

end RefreshTopProof.Part_I
