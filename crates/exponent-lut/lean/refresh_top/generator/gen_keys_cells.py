"""Write `Part_<c_j>_KeysCells.lean`: the keys of every online PRF cell are the keys the
preprocessing computes for it.

Usage: `python3 gen_keys_cells.py <parameter directory>`.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(__file__))
from gen_part import parts_of
from patterns import destruct_inline
from layouts import layouts


def prefixed(text, p='p'):
    text = re.sub(r'\bw_(\d)', rf'{p}w_\1', text)
    text = re.sub(r'\bh(\d)', rf'{p}h\1', text)
    text = re.sub(r'\bhout\b', f'{p}hout', text)
    return text


def cell0(part, on, pre, QA, QB, MA, MB, MC, LB, HB, layA, HA, name):
    """Bucket 0: label at `QA` (gadget `MA`), one encoding `MB` columns, RHS at `QB` (`MC` columns)."""
    d = sys.argv[1]
    ons = f'Stage_online_eval_{part}'
    prs = f'Stage_preprocess_{part}'
    don = destruct_inline(f'{d}/generated/{ons}.lean', ons, f'parallel_generatedRoot_{on}', 'hon')
    dpre = prefixed(destruct_inline(f'{d}/generated/{prs}.lean', prs, f'parallel_generatedRoot_{pre}', 'hpre'))
    return f'''
set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `0` of the {name} PRF (online `{on}`) are those the
preprocessing computes (`{pre}`). -/
theorem keysCell{on} {{model : HashModel}} {{params : {ons}.Params}} {{params' : {prs}.Params}}
    {{key : ByteArray}} {{i : Nat}}
    {{ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix {QA} N 1 {MA}}} {{apay : Fin 5 → ExactMatrix {QA} N 1 1}}
    {{As00 As10 As01 As11 cs0 cs1 : ExactMatrix {QA} N 1 {MB}}} {{bot0 bot1 : ExactMatrix {QB} N 1 {MC}}}
    {{outs : Blocks {QB} {MC}}}
    (hon : {ons}.parallel_generatedRoot_{on} Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {{sec one : ExactMatrix {QA} N 1 1}} {{s e : Int}} {{pouts : _}}
    (hpre : {prs}.parallel_generatedRoot_{pre} Backend.backend model params' i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample {QA} N 1 {MA} key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
{don}{dpre}  -- The label members.
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
  obtain ⟨t20, b20⟩ := decompose_stack Backend.{layA} {HA} rfl h18_0 h19_0 ph5_0 ph11_0
  have r20 := rebase_self (by norm_num) h20_0
  obtain ⟨t35, b35⟩ := decompose_stack Backend.{layA} {HA} rfl h33_0 h34_0 ph22_0 ph25_0
  have r35 := rebase_self (by norm_num) h35_0
  obtain ⟨-, s21, s21'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_20_0
  obtain ⟨-, s36, s36'⟩ := stacked_row h4_0_1 h7_0 h11_0_1 h14_0 h15_0 w_35_0
  obtain ⟨p3a, p3b⟩ := stack2_mul_rows ph3_0 pw_5_0
  obtain ⟨p9a, p9b⟩ := stack2_mul_rows ph9_0 pw_11_0
  obtain ⟨p3a', p3b'⟩ := stack2_mul_rows ph3_0 pw_22_0
  obtain ⟨p9a', p9b'⟩ := stack2_mul_rows ph9_0 pw_25_0
  have k28 : w_28_0 = pw_14_0 := by
    rw [slice_rowOf (by norm_num) h28_0, slice_rowOf (by norm_num) ph14_0]
    show rowOf (w_15_0 * w_20_0) 1 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 0
    rw [s21, pairMul, rowOf_add, p3a, p9a, r20, t20, b20, e1, e7]
    rfl
  have k29 : w_29_0 = pw_15_0 := by
    rw [slice_rowOf (by norm_num) h29_0, slice_rowOf (by norm_num) ph15_0]
    show rowOf (w_15_0 * w_20_0) 2 = rowOf (pw_3_0 * pw_5_0 + pw_9_0 * pw_11_0) 1
    rw [s21', pairMul, rowOf_add, p3b, p9b, r20, t20, b20, e2, e8]
    rfl
  have k41 : w_41_0 = pw_28_0 := by
    rw [slice_rowOf (by norm_num) h41_0, slice_rowOf (by norm_num) ph28_0]
    show rowOf (w_15_0 * w_35_0) 1 = rowOf (pw_3_0 * pw_22_0 + pw_9_0 * pw_25_0) 0
    rw [s36, pairMul, rowOf_add, p3a', p9a', r35, t35, b35, e1, e7]
    rfl
  have k42 : w_42_0 = pw_29_0 := by
    rw [slice_rowOf (by norm_num) h42_0, slice_rowOf (by norm_num) ph29_0]
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
  obtain ⟨t50, b50⟩ := decompose_stack Backend.{LB} {HB} rfl h48_0 h49_0 ph18_0 ph48_0
  have t50' := rebase_topRows h50_0 ph19_0 t50
  have b50' := rebase_bottomRows h50_0 ph49_0 b50
  obtain ⟨t58, b58⟩ := decompose_stack Backend.{LB} {HB} rfl h56_0 h57_0 ph55_0 ph67_0
  have t58' := rebase_topRows h58_0 ph56_0 t58
  have b58' := rebase_bottomRows h58_0 ph68_0 b58
  obtain ⟨-, s51, s51'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_50_0
  obtain ⟨-, s59, s59'⟩ := stacked_row h28_0_1 h30_0 h41_0_1 h43_0 h44_0 w_58_0
  obtain ⟨q16a, q16b⟩ := stack2_mul_rows ph16_0 pw_19_0
  obtain ⟨q30a, q30b⟩ := stack2_mul_rows ph30_0 pw_49_0
  obtain ⟨q16a', q16b'⟩ := stack2_mul_rows ph16_0 pw_56_0
  obtain ⟨q30a', q30b'⟩ := stack2_mul_rows ph30_0 pw_68_0
  have k52 : w_52_0 = pw_52_0 := by
    rw [slice_rowOf (by norm_num) h52_0, slice_rowOf (by norm_num) ph52_0]
    show rowOf (w_44_0 * w_50_0) 1 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 0
    rw [s51, pairMul, rowOf_add, q16a, q30a, t50', b50', k28, k41]
    rfl
  have k62 : w_62_0 = pw_73_0 := by
    rw [slice_rowOf (by norm_num) h62_0, slice_rowOf (by norm_num) ph73_0]
    show rowOf (w_44_0 * w_50_0) 2 = rowOf (pw_16_0 * pw_19_0 + pw_30_0 * pw_49_0) 1
    rw [s51', pairMul, rowOf_add, q16b, q30b, t50', b50', k29, k42]
    rfl
  have k60 : w_60_0 = pw_71_0 := by
    rw [slice_rowOf (by norm_num) h60_0, slice_rowOf (by norm_num) ph71_0]
    show rowOf (w_44_0 * w_58_0) 1 = rowOf (pw_16_0 * pw_56_0 + pw_30_0 * pw_68_0) 0
    rw [s59, pairMul, rowOf_add, q16a', q30a', t58', b58', k28, k41]
    rfl
  have k64 : w_64_0 = pw_75_0 := by
    rw [slice_rowOf (by norm_num) h64_0, slice_rowOf (by norm_num) ph75_0]
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
'''


def cell1(part, on, pre, QA, QB, QC, MA, MB, MC, MD, LA, HA, LB, HB, LC, HC, P, name, members, cells):
    """Bucket 1: label at `QA` (gadget `MA`) aligned down to `QB` by `D(p G)` (`MB` digits), the
    bucket-0 members at `QB` (`MC` columns), the RHS at `QC` (`MD` columns)."""
    d = sys.argv[1]
    ons = f'Stage_online_eval_{part}'
    prs = f'Stage_preprocess_{part}'
    don = destruct_inline(f'{d}/generated/{ons}.lean', ons, f'parallel_generatedRoot_{on}', 'hon')
    dpre = prefixed(destruct_inline(f'{d}/generated/{prs}.lean', prs, f'parallel_generatedRoot_{pre}', 'hpre'))
    return f"""
set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
/-- The keys of cell `i` of bucket `1` of the {name} PRF (online `{on}`) are those the
preprocessing computes (`{pre}`). -/
theorem keysCell{on} {{model : HashModel}} {{params : {ons}.Params}} {{params' : {prs}.Params}}
    {{key : ByteArray}} {{i : Nat}}
    {{ca0 ca1 aA00 aA10 aA01 aA11 : Fin 5 → ExactMatrix {QA} N 1 {MA}}} {{apay : Fin 5 → ExactMatrix {QA} N 1 1}}
    {{As00 As10 As01 As11 cs0 cs1 : Fin {members} → ExactMatrix {QB} N 1 {MC}}}
    {{bot0 bot1 : ExactMatrix {QC} N 1 {MD}}} {{outs : Blocks {QC} {MD}}}
    (hon : {ons}.parallel_generatedRoot_{on} Backend.backend model params i
      (ca0, aA00, aA10, ca1, aA01, aA11, As00, As10, cs0, apay, As01, As11, cs1, key, bot0, bot1, ())
      outs)
    {{sec one : ExactMatrix {QA} N 1 1}} {{s e : Int}} {{pouts : _}}
    (hpre : {prs}.parallel_generatedRoot_{pre} Backend.backend model params' i
      (key, As00, As10, As01, As11, sec, s, e, one, ()) pouts)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![aA00, aA01, aA10, aA11] c k =
      model.sample {QA} N 1 {MA} key (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)]))
    (hbot0 : bot0 = pouts.2.2.2.2.2.2.1) (hbot1 : bot1 = pouts.2.2.2.2.2.2.2.1) :
    outs.1 = pouts.1 ∧ outs.2.1 = pouts.2.1 ∧ outs.2.2.1 = pouts.2.2.1 ∧
      outs.2.2.2.1 = pouts.2.2.2.1 := by
{don}{dpre}  -- The label members.
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
  obtain ⟨t45, b45⟩ := decompose_stack Backend.{LB} {HB} rfl h43_0 h44_0 ph16_0 ph27_0
  have r45 := rebase_self (by norm_num) h45_0
  obtain ⟨t64, b64⟩ := decompose_stack Backend.{LB} {HB} rfl h62_0 h63_0 ph39_0 ph43_0
  have r64 := rebase_self (by norm_num) h64_0
  obtain ⟨-, s46, s46'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_45_0
  obtain ⟨-, s65, s65'⟩ := stacked_row h18_0_1 h23_0 h31_0_1 h36_0 h37_0 w_64_0
  obtain ⟨p12a, p12b⟩ := stack2_mul_rows ph12_0 pw_16_0
  obtain ⟨p24a, p24b⟩ := stack2_mul_rows ph24_0 pw_27_0
  obtain ⟨p12a', p12b'⟩ := stack2_mul_rows ph12_0 pw_39_0
  obtain ⟨p24a', p24b'⟩ := stack2_mul_rows ph24_0 pw_43_0
  have k55 : w_55_0 = pw_30_0 := by
    rw [slice_rowOf (by norm_num) h55_0, slice_rowOf (by norm_num) ph30_0]
    show rowOf (w_37_0 * w_45_0) 1 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 0
    rw [s46, pairMul, rowOf_add, p12a, p24a, r45, t45, b45, a18, a31]
    rfl
  have k56 : w_56_0 = pw_31_0 := by
    rw [slice_rowOf (by norm_num) h56_0, slice_rowOf (by norm_num) ph31_0]
    show rowOf (w_37_0 * w_45_0) 2 = rowOf (pw_12_0 * pw_16_0 + pw_24_0 * pw_27_0) 1
    rw [s46', pairMul, rowOf_add, p12b, p24b, r45, t45, b45, a22, a35]
    rfl
  have k71 : w_71_0 = pw_46_0 := by
    rw [slice_rowOf (by norm_num) h71_0, slice_rowOf (by norm_num) ph46_0]
    show rowOf (w_37_0 * w_64_0) 1 = rowOf (pw_12_0 * pw_39_0 + pw_24_0 * pw_43_0) 0
    rw [s65, pairMul, rowOf_add, p12a', p24a', r64, t64, b64, a18, a31]
    rfl
  have k72 : w_72_0 = pw_47_0 := by
    rw [slice_rowOf (by norm_num) h72_0, slice_rowOf (by norm_num) ph47_0]
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
  obtain ⟨t80, b80⟩ := decompose_stack Backend.{LC} {HC} rfl h78_0 h79_0 ph34_0 ph66_0
  have t80' := rebase_topRows h80_0 ph35_0 t80
  have b80' := rebase_bottomRows h80_0 ph67_0 b80
  obtain ⟨t88, b88⟩ := decompose_stack Backend.{LC} {HC} rfl h86_0 h87_0 ph73_0 ph85_0
  have t88' := rebase_topRows h88_0 ph74_0 t88
  have b88' := rebase_bottomRows h88_0 ph86_0 b88
  obtain ⟨-, s81, s81'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_80_0
  obtain ⟨-, s89, s89'⟩ := stacked_row h55_0_1 h57_0 h71_0_1 h73_0 h74_0 w_88_0
  obtain ⟨q32a, q32b⟩ := stack2_mul_rows ph32_0 pw_35_0
  obtain ⟨q48a, q48b⟩ := stack2_mul_rows ph48_0 pw_67_0
  obtain ⟨q32a', q32b'⟩ := stack2_mul_rows ph32_0 pw_74_0
  obtain ⟨q48a', q48b'⟩ := stack2_mul_rows ph48_0 pw_86_0
  have k82 : w_82_0 = pw_70_0 := by
    rw [slice_rowOf (by norm_num) h82_0, slice_rowOf (by norm_num) ph70_0]
    show rowOf (w_74_0 * w_80_0) 1 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 0
    rw [s81, pairMul, rowOf_add, q32a, q48a, t80', b80', k55, k71]
    rfl
  have k92 : w_92_0 = pw_91_0 := by
    rw [slice_rowOf (by norm_num) h92_0, slice_rowOf (by norm_num) ph91_0]
    show rowOf (w_74_0 * w_80_0) 2 = rowOf (pw_32_0 * pw_35_0 + pw_48_0 * pw_67_0) 1
    rw [s81', pairMul, rowOf_add, q32b, q48b, t80', b80', k56, k72]
    rfl
  have k90 : w_90_0 = pw_89_0 := by
    rw [slice_rowOf (by norm_num) h90_0, slice_rowOf (by norm_num) ph89_0]
    show rowOf (w_74_0 * w_88_0) 1 = rowOf (pw_32_0 * pw_74_0 + pw_48_0 * pw_86_0) 0
    rw [s89, pairMul, rowOf_add, q32a', q48a', t88', b88', k55, k71]
    rfl
  have k94 : w_94_0 = pw_93_0 := by
    rw [slice_rowOf (by norm_num) h94_0, slice_rowOf (by norm_num) ph93_0]
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
"""


def cell_pairs(d, part):
    """The online cell parallels and their preprocessing counterparts (by hash key source)."""
    sys.path.insert(0, os.path.dirname(__file__))
    from patterns import parse
    from wiring import stage_args, root_inputs
    def keymap(stage, keypos):
        p = f'{d}/generated/{stage}.lean'
        _, n, a = parse(p, 'generatedRoot')
        ins = root_inputs(p)
        args, _ = stage_args(d, stage)
        src = {w: args[i] for i, w in enumerate(ins)}
        out = []
        for y in a:
            m = re.match(r'\(∀ i : Fin (40|240|800), parallel_generatedRoot_(\d+) backend hashModel params i \((.*)', y)
            if m:
                wires = re.findall(r'\bw_\d+_\d+\b', m.group(3))
                out.append((int(m.group(2)), int(m.group(1)), src[wires[keypos]]))
        return out
    on = keymap(f'Stage_online_eval_{part}', 13)
    pre = keymap(f'Stage_preprocess_{part}', 0)
    pairs = []
    for idx, (k, cells, src) in enumerate(on):
        ks = [pk for pk, pc, ps in pre if ps == src and pc == cells]
        bucket = 0 if idx % 2 == 0 else 1
        pairs.append((k, ks[bucket], cells, bucket))
    return pairs


d = sys.argv[1]
L = layouts(d)
for part in parts_of(d):
    out = [f'''import TopKeys
import Part_{part}_Cells
import ExponentFunctional

/-! The keys of the online PRF cells (written by `generator/gen_keys_cells.py`). -/

namespace RefreshTopProof.Part_{part}

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

/-- `rns-bgg/input/WireId(1)/0`, the hash tag of the label's public matrices. -/
def labelPrefix : Blob := [114, 110, 115, 45, 98, 103, 103, 47, 105, 110, 112, 117, 116, 47, 87,
  105, 114, 101, 73, 100, 40, 49, 41, 47, 48]
''']
    for on, pre, cells, bucket in cell_pairs(d, part):
        name = {40: 'mask', 240: 'fresh', 800: 'smudge'}[cells]
        members = cells // 5
        if cells != 800 and bucket == 0:
            out.append(cell0(part, on, pre, 'Q7', 'Q6', 14, 12, 10, L['Q6'], 'layoutQ6', L['Q7'],
                             'layoutQ7', name))
        elif cells != 800:
            out.append(cell1(part, on, pre, 'Q7', 'Q6', 'Q5', 14, 12, 10, 8, L['Q7'], 'layoutQ7',
                             L['Q6'], 'layoutQ6', L['Q5'], 'layoutQ5', 'p6', name, members, cells))
        elif bucket == 0:
            out.append(cell0(part, on, pre, 'Q6', 'Q5', 12, 10, 8, L['Q5'], 'layoutQ5', L['Q6'],
                             'layoutQ6', name))
        else:
            out.append(cell1(part, on, pre, 'Q6', 'Q5', 'Q0', 12, 10, 8, 6, L['Q6'], 'layoutQ6',
                             L['Q5'], 'layoutQ5', L['Q0'], 'layoutQ0', 'p5', name, members, cells))
    out.append(f'\nend RefreshTopProof.Part_{part}\n')
    open(f'{d}/Part_{part}_KeysCells.lean', 'w').write(''.join(out))
    break
