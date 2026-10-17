#!/usr/bin/env python3
"""Derive the runtime lemmas of online steps 1-3 from those of step 0.

Steps 1 and 2 run the scopes of step 0 under other ids with the LUT table at offset 128 * step.
Step 3 reads its table as constant coefficients (`liftInteger`) instead of monomials, so its
coefficient terms are `coefTermC` and its coefficient loops destructure one witness less.
"""
import re
from pathlib import Path
from textwrap import dedent

HERE = Path(__file__).resolve().parent
SCOPES = {
    step: dict(zip([23, 33, 39, 58, 65, 75, 82, 92, 99, 109, 116, 126, 133],
                   [149 + 119 * (step - 1), 157 + 119 * (step - 1),
                    162 + 119 * (step - 1)] +
                   [root + 119 * step for root in [58, 65, 75, 82, 92, 99, 109, 116, 126, 133]]))
    for step in (1, 2, 3)
}
GROUP_ROOTS = [75, 92, 109, 126]
OUTER_SCOPES = [39, 46, 46, 46]
SLOT_SCOPES = [[43, 50, 55, 60]] + [[50, 55, 60, 65] for _ in range(3)]
SLOT_TEMPLATE = r"""
set_option maxHeartbeats 4000000 in
theorem slotCol«slot»_16 {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_online_eval.Params} {i0 tick : Nat} {c top bot : ExactMatrix Q N 1 88} {i : Nat} (hi : i < 88)
    {s t : ExactMatrix Q N 1 88}
    (hb : Stage_online_eval.sequential_sequential_parallel_generatedRoot_99_«scope»_16 Backend.backend model tape path params i0 tick i
      (s, (c, (top, (bot, ())))) t) :
    t = s + (c * rhsDigits top bot ⟨i, hi⟩) 0 0 • (unitRow ((i : Nat) : Int) : ExactMatrix Q N 1 88) ∧
      ColumnRuns top bot ⟨i, hi⟩ := by
  obtain ⟨w3, w5, w6, w7, w8, w11, -, -, -, -, -, -, x3, -, -, -, -, -, -, x5, x6, x7, x8, -, -,
    -, -, -, -, -, -, x11, xout⟩ := hb
  have hw11 := unitRow_window (c := Int.ofNat i) (by omega) x11
  obtain ⟨_, hw3⟩ := sliceMatrix_column x3
  obtain ⟨_, hw5⟩ := sliceMatrix_column x5
  simp only [Int.ofNat_eq_natCast, Int.toNat_natCast] at hw3 hw5 hw11
  rw [concatRows_pairColumn x6, hw3, hw5] at x7
  have hrun := x7
  have hw8 := columnDigits_eq x7 x8
  subst xout hw8 hw11
  refine ⟨?_, ⟨w7, hrun, x8⟩⟩
  simp only [matrixAdd, matrixMulScalarLeft_eq, matrixMul]

set_option maxHeartbeats 4000000 in
theorem slotCol«slot»_42 {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_online_eval.Params} {i0 tick : Nat} {c top bot : ExactMatrix Q N 1 88} {i : Nat} (hi : i < 88)
    {s t : ExactMatrix Q N 1 88}
    (hb : Stage_online_eval.sequential_sequential_parallel_generatedRoot_99_«scope»_42 Backend.backend model tape path params i0 tick i
      (s, (c, (top, (bot, ())))) t) :
    t = s + (c * rhsDigits top bot ⟨i, hi⟩) 0 0 • (unitRow ((i : Nat) : Int) : ExactMatrix Q N 1 88) ∧
      ColumnRuns top bot ⟨i, hi⟩ := by
  obtain ⟨w3, w5, w6, w7, w8, w11, -, -, -, -, -, -, x3, -, -, -, -, -, -, x5, x6, x7, x8, -, -,
    -, -, -, -, -, -, x11, xout⟩ := hb
  have hw11 := unitRow_window (c := Int.ofNat i) (by omega) x11
  obtain ⟨_, hw3⟩ := sliceMatrix_column x3
  obtain ⟨_, hw5⟩ := sliceMatrix_column x5
  simp only [Int.ofNat_eq_natCast, Int.toNat_natCast] at hw3 hw5 hw11
  rw [concatRows_pairColumn x6, hw3, hw5] at x7
  have hrun := x7
  have hw8 := columnDigits_eq x7 x8
  subst xout hw8 hw11
  refine ⟨?_, ⟨w7, hrun, x8⟩⟩
  simp only [matrixAdd, matrixMulScalarLeft_eq, matrixMul]

set_option maxHeartbeats 4000000 in
theorem slotCoef«slot» {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_online_eval.Params} {tables : Fin 512 → Int} {b tick i : Nat} (hb : b < 7)
    (hi : i < 128) {s t : ExactMatrix Q N 1 1}
    (h : Stage_online_eval.sequential_sequential_parallel_generatedRoot_99_«scope»_51 Backend.backend model tape path
      params b tick i (s, (tables, ())) t) :
    t = s + coefTerm tables 0 (kSlot «slot» b) ⟨i, hi⟩ (by norm_num) := by
  obtain ⟨w6, w16, -, -, x6, -, -, -, -, x16, xout⟩ := h
  have hsel : «coefSelect» = Int.ofNat (kSlot «slot» b) := by
    interval_cases b <;> rfl
  rw [hsel] at x16
  obtain ⟨_, hw6⟩ := familyGet_eq x6
  obtain ⟨-, -, -, hw16⟩ := x16
  subst xout
  simp only [coefTerm, hw16]
  congr 3
  rw [hw6]

set_option maxHeartbeats 64000000 in
theorem slotFinish«slot» {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_online_eval.Params} {i tick : Nat} (hi : i < 7) (d : BranchData)
    {E : ExactMatrix Q N 1 88} (hE : E = d.Eb (15 + i) (by omega))
    {next0 next1 : Fin 7 → ExactMatrix Q N 1 88}
    {tables : Fin 512 → Int} {s t : LeafState}
    (hT : s.2.2.1 = d.TL0 ⟨i, hi⟩ ∧ s.2.2.2.1 = d.TL1 ⟨i, hi⟩)
    (h : Stage_online_eval.sequential_parallel_generatedRoot_99_«scope» Backend.backend model tape path params i tick
      (s.1, (s.2.1, (s.2.2.1, (s.2.2.2.1, (d.hf0, (d.hf1, (d.c0, (d.c1, (d.key, (E, (tables, (next0, (next1, ()))))))))))))) t) :
    d.Runs model (15 + i) (by omega) ∧
      t.1 = s.1 + d.row model (15 + i) (by omega) 0 * scalarV (branchCoef tables 0 (kSlot «slot» i) (by norm_num)) ∧
      t.2.1 = s.2.1 + d.row model (15 + i) (by omega) 1 *
        scalarV (branchCoef tables 0 (kSlot «slot» i) (by norm_num)) ∧
      t.2.2.1 = next0 ⟨i, hi⟩ ∧
      t.2.2.2.1 = next1 ⟨i, hi⟩ := by
  have hb : 15 + i < 22 := by omega
  obtain ⟨w, hbody⟩ := h
  let ctx : Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».Ctx Backend.backend model tape path params i tick :=
    ⟨(s.1, s.2.1, s.2.2.1, s.2.2.2.1, d.hf0, d.hf1, d.c0, d.c1, d.key, E, tables, next0, next1, ()), w⟩
  have F := Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».facts ctx hbody
  have hsel := leafSwitchSelect i hi
  obtain ⟨hT0, hT1⟩ := hT
  have eT0 : s.2.2.1 = d.Tb0 (15 + i) hb := hT0.trans (d.Tb0_leaf hi).symm
  have eT1 : s.2.2.2.1 = d.Tb1 (15 + i) hb := hT1.trans (d.Tb1_leaf hi).symm
  -- The helper of `k`.
  have hw3 : Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_3_0 ctx = Int.ofNat (15 + i) := leafMemberSelect i hi
  have h4 := F.c0.h4_0
  have h6 := F.c0.h6_0
  rw [hw3] at h4 h6
  obtain ⟨_, hw4⟩ := familyGet_eq h4
  obtain ⟨_, hw6⟩ := familyGet_eq h6
  have hw7 : Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_7_0 ctx = joinRow (d.hrow (15 + i) hb) := by
    have h7 := F.c0.h7_0
    rw [hw4, hw6] at h7
    exact concat_joinRow (r := d.hrow (15 + i) hb) h7
  -- The key `T`, rejoined from its halves.
  have hw11 : Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_11_0 ctx = d.Tb0 (15 + i) hb := by
    have h9 := sliceMatrix_window 0 F.c0.h9_0 (by simp) (by omega)
    have h10 := sliceMatrix_window 44 F.c0.h10_0 (by simp) (by omega)
    have h11 := F.c0.h11_0
    rw [h9, h10] at h11
    rw [(concat_joinRow (r := halves _) h11).trans (joinRow_halves _)]
    exact eT0
  have hw15 : Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_15_0 ctx = d.Tb1 (15 + i) hb := by
    have h13 := sliceMatrix_window 0 F.c0.h13_0 (by simp) (by omega)
    have h14 := sliceMatrix_window 44 F.c1.h14_0 (by simp) (by omega)
    have h15 := F.c1.h15_0
    rw [h13, h14] at h15
    rw [(concat_joinRow (r := halves _) h15).trans (joinRow_halves _)]
    exact eT1
  -- The helper action.
  have hcount : Int.toNat 88 = 88 := rfl
  have h16 := F.c1.h16_0
  rw [hw7, hw11, hw15, hcount] at h16
  have hact := colAction (r := 1) (fun _ (s : ExactMatrix Q N 1 88) ↦ s)
    (rows := ![joinRow (d.hrow (15 + i) hb)]) (fun _ ↦ rfl)
    (fun i' s t hi' hb' ↦ by
      obtain ⟨e, hrun⟩ := slotCol«slot»_16 hi' hb'
      exact ⟨fun r ↦ by fin_cases r; exact e, hrun⟩) h16
  have hhelp := colActionSplit (c := joinRow (d.hrow (15 + i) hb)) (row := d.hrow (15 + i) hb)
    (concatColumns_joinRow _) (tk := halves (d.Tb0 (15 + i) hb)) (bk := halves (d.Tb1 (15 + i) hb))
    (entry_halves _) (entry_halves _) (acc := Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_16_0 ctx)
    (fun jj ↦ ⟨(hact jj).1 0, (hact jj).2⟩)
  -- The routed state rows.
  have hroute : ∀ {x y : ExactMatrix Q N 1 44},
      ringAutomorphismRuns (MxxIR.selectInt (Int.ofNat i) leafSwitches) x y →
      y = substituteMatrix (kSw (15 + i)) (kSw_odd ⟨15 + i, hb⟩) x := by
    intro x y hxy
    rw [hsel] at hxy
    rw [Int.ofNat_eq_natCast, kSw_cast (15 + i) hb] at hxy
    exact routing_kSw hb hxy
  have hw23 : Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_23_0 ctx = joinRow (d.σc (15 + i) hb) := by
    have h23 := F.c1.h23_0
    rw [hroute F.c1.h20_0, hroute F.c1.h22_0] at h23
    exact concat_joinRow (r := d.σc (15 + i) hb) h23
  -- The switching ciphertext `C_k`.
  have hkic := kInv_cast (15 + i) hb
  have hHi : ∀ {c : Nat} (hc : c < 4) {y : ExactMatrix Q N 1 44},
      hashSample model dynamicPrefix [.bytes schemaBytes, .bytes helperKeyBytes,
        .u64Le (MxxIR.selectInt (Int.ofNat i) leafInvSwitches), .bytes componentBytes, .u64Le (c : Int)]
        d.key y → y = helperKey model d.key (kInv (15 + i)) ⟨c / 2, by omega⟩ ⟨c % 2, by omega⟩ := by
    intro c hc y hy
    rw [leafInvSwitchSelect i hi] at hy
    exact helperKey_read hc hy
  have hw29 : Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_29_0 ctx = joinRow (d.top model (15 + i) hb) := by
    have h29 := F.c2.h29_0
    rw [hroute F.c1.h26_0, hroute F.c2.h28_0, hHi (c := 0) (by norm_num) F.c1.h25_0,
      hHi (c := 1) (by norm_num) F.c1.h27_0] at h29
    exact concat_joinRow (r := d.top model (15 + i) hb) h29
  have hw32 : Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_32_0 ctx = Int.ofNat (36 - (15 + i)) := leafInvMemberSelect i hi
  have h33 := F.c2.h33_0
  have h38 := F.c2.h38_0
  rw [hw32] at h33 h38
  obtain ⟨_, hw33⟩ := familyGet_eq h33
  obtain ⟨_, hw38⟩ := familyGet_eq h38
  have hw41 : Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_41_0 ctx = joinRow (d.bot model (15 + i) hb) := by
    have h41 := F.c2.h41_0
    have hw35 : Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_35_0 ctx =
        matrixSub (Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_31_0 ctx)
          (Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_34_0 ctx) := rfl
    have hw40 : Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_40_0 ctx =
        matrixSub (Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_37_0 ctx)
          (Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_39_0 ctx) := rfl
    rw [hw35, hw40, hroute F.c2.h31_0, hroute F.c2.h37_0, hroute F.c2.h34_0, hroute F.c2.h39_0,
      hHi (c := 2) (by norm_num) F.c2.h30_0, hHi (c := 3) (by norm_num) F.c2.h36_0, hw33, hw38] at h41
    exact concat_joinRow (r := d.bot model (15 + i) hb) h41
  -- The switch.
  have h42 := F.c3.h42_0
  rw [hw23, hw29, hw41, hcount] at h42
  have hsw := colAction (r := 1) (fun _ (s : ExactMatrix Q N 1 88) ↦ s)
    (rows := ![joinRow (d.σc (15 + i) hb)]) (fun _ ↦ rfl)
    (fun i' s t hi' hb' ↦ by
      obtain ⟨e, hrun⟩ := slotCol«slot»_42 hi' hb'
      exact ⟨fun r ↦ by fin_cases r; exact e, hrun⟩) h42
  have hswitch := colActionSplit (c := joinRow (d.σc (15 + i) hb)) (row := d.σc (15 + i) hb)
    (concatColumns_joinRow _) (tk := d.top model (15 + i) hb) (bk := d.bot model (15 + i) hb)
    (fun _ ↦ rfl) (fun _ ↦ rfl) (acc := Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_42_0 ctx)
    (fun jj ↦ ⟨(hsw jj).1 0, (hsw jj).2⟩)
  have hw53 : Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_44_0 ctx =
      d.Eb (15 + i) hb := hE
  -- The coefficient and the scalar action.
  have h60 := F.c3.h51_0
  have hc128 : Int.toNat 128 = 128 := rfl
  rw [hc128] at h60
  have hcoef := coefAccum (by norm_num) (fun i' s t hi' hb' ↦ slotCoef«slot» hi hi' hb') h60
  have hV := scalarV_eq F.c3.h53_0 (coef := branchCoef tables 0 (kSlot «slot» i) (by norm_num)) (by
    have h64 := F.c3.h55_0
    have : Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_51_0 ctx =
        branchCoef tables 0 (kSlot «slot» i) (by norm_num) := hcoef
    change gadgetDecomposeRuns _ _ _ (matrixMulScalarLeft (matrixMulScalarLeft winvPoly
      (Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_51_0 ctx))
        (Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_53_0 ctx)) _ at h64
    rw [this] at h64
    exact h64)
  -- The next leaf's key.
  have hnext : Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_67_0 ctx =
      Int.ofNat i := rfl
  have h77 := F.c4.h68_0
  have h79 := F.c4.h70_0
  rw [hnext] at h77 h79
  obtain ⟨_, hw77⟩ := familyGet_eq h77
  obtain ⟨_, hw79⟩ := familyGet_eq h79
  refine ⟨⟨hhelp.2, hswitch.2⟩, ?_, ?_, ?_, ?_⟩
  · rw [F.c4.hout]
    have a0 := hhelp.1 0
    have b0 := hswitch.1 0
    simp only [Fin.val_zero] at a0 b0
    have h17 := sliceMatrix_window 0 F.c1.h17_0 (by simp) (by omega)
    have h43 := sliceMatrix_window 0 F.c3.h43_0 (by simp) (by omega)
    have h54 := sliceMatrix_window 0 F.c3.h45_0 (by simp) (by omega)
    rw [hw53] at h54
    show matrixAdd s.1 (matrixMul (matrixAdd (Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_17_0 ctx)
      (matrixAdd (Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_43_0 ctx)
        (Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_45_0 ctx)))
      (Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_55_0 ctx)) = _
    rw [h17, h43, h54, a0, b0, hV]
    rfl
  · rw [F.c4.hout]
    have a1 := hhelp.1 1
    have b1 := hswitch.1 1
    simp only [Fin.val_one] at a1 b1
    have h68 := sliceMatrix_window 44 F.c3.h59_0 (by simp) (by omega)
    have h69 := sliceMatrix_window 44 F.c4.h60_0 (by simp) (by omega)
    have h70 := sliceMatrix_window 44 F.c4.h61_0 (by simp) (by omega)
    rw [hw53] at h70
    show matrixAdd s.2.1 (matrixMul (matrixAdd (Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_59_0 ctx)
      (matrixAdd (Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_60_0 ctx)
        (Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_61_0 ctx)))
      (Stage_online_eval.sequential_parallel_generatedRoot_99_«scope».w_55_0 ctx)) = _
    rw [h68, h69, h70, a1, b1, hV]
    rfl
  · rw [F.c4.hout]
    exact hw77
  · rw [F.c4.hout]
    exact hw79

"""


# Step 3 coefficient-loop proofs: the term is `liftInteger t` and the witnesses shift by one.
FINAL_COEF = [
    ("obtain ⟨w6, w16, -, -, x6, -, -, -, -, x16, xout⟩ := h", "obtain ⟨w5, w13, -, -, x5, -, -, -, -, x13, xout⟩ := h"),
    ("obtain ⟨w6, w16, -, -, x6, x16, xout⟩ := hb", "obtain ⟨w5, w13, -, -, x5, x13, xout⟩ := hb"),
    ("obtain ⟨w6, w16, -, -, x6, -, -, x16, xout⟩ := h", "obtain ⟨w5, w13, -, -, x5, -, -, x13, xout⟩ := h"),
    ("rw [hsel] at x16", "rw [hsel] at x13"),
    ("obtain ⟨_, hw6⟩ := familyGet_eq x6", "obtain ⟨_, hw5⟩ := familyGet_eq x5"),
    ("obtain ⟨-, -, -, hw16⟩ := x16", "obtain ⟨-, -, -, hw13⟩ := x13"),
    ("simp only [dif_pos hi, coefTerm, hw16]", "simp only [dif_pos hi, coefTermC, hw13]"),
    ("simp only [coefTerm, hw16]", "simp only [coefTermC, hw13]"),
    ("congr 3\n      rw [hw6]", "congr 2\n      rw [hw5]"),
    ("congr 3\n  rw [hw6]", "congr 2\n  rw [hw5]"),
]


def final_coef(match: re.Match) -> str:
    text = match.group(0)
    for old, new in FINAL_COEF:
        text = text.replace(old, new)
    return text


def derive(source: str, step: int) -> str:
    out = source
    out = re.sub(r"generatedRoot_(\d+)(?=[_ .\n)`])", lambda match:
        "generatedRoot_" + str(SCOPES[step].get(int(match[1]), int(match[1]))), out)
    prefix, grouped = out.split('/-! ## Member-group slot bodies -/', 1)
    def rename_path(match: re.Match, local: bool = False) -> str:
        index = int(match[1])
        mapped = index if local and index == 39 else SCOPES[step].get(index, index)
        return f'[{mapped}, '
    out = (re.sub(r"\[(\d+), ", rename_path, prefix) + '/-! ## Member-group slot bodies -/' +
           re.sub(r"\[(\d+), ", lambda match: rename_path(match, local=True), grouped))
    out = out.replace("namespace PeProof.Step0", f"namespace PeProof.Step{step}")
    out = out.replace("end PeProof.Step0", f"end PeProof.Step{step}")
    out = out.replace("# Online evaluation, step `0`", f"# Online evaluation, step `{step}`")
    out = re.sub(r"(coefTerm|branchCoef) tables 0 ", rf"\1 tables {128 * step} ", out)
    if step == 3:
        out = re.sub(r"(?ms)^theorem (?:coefLoop0|coefStep\d+_\d+|slotCoef\d+) .*?(?=\n\n)", final_coef, out)
        out = re.sub(r"\bcoefTerm\b", "coefTermC", out)
        out = re.sub(r"\bbranchCoef\b", "branchCoefC", out)
        out = re.sub(r"\bcoefAccum\b", "coefAccumC", out)
    header = (f"-- Generated by `gen_steps.py` from `PeOnlineStep0.lean`; edit the source instead.\n")
    return header + out


def projection(name, index):
    return name + '.2' * index + '.1'

def tuple_of(items):
    return '(' + ', '.join(items + ['()']) + ')'

def array_of(items):
    return '![' + ', '.join(items) + ']'

def build_shared(group: int) -> str:
    root, outer_id = GROUP_ROOTS[group], OUTER_SCOPES[group]
    rows = [projection('s', slot) for slot in range(4)]
    state_type = ' × '.join(['Mat88'] * 4 + ['Unit'])
    block_type = ' × '.join(['Mat88'] * 4 + ['Mat44', 'Mat44', 'Unit'])
    shared = f'''

    abbrev Mat88 := ExactMatrix Q N 1 88
    abbrev Mat44 := ExactMatrix Q N 1 44
    abbrev Rows4 := {state_type}
    abbrev SharedBlock := {block_type}
    noncomputable abbrev rows4 (s : Rows4) : Fin 4 → Mat88 := {array_of(rows)}
    noncomputable abbrev blockRows (s : SharedBlock) : Fin 4 → Mat88 := {array_of(rows)}
    noncomputable abbrev blockProjection (s : SharedBlock) (slot : Fin 4) : MulBlock :=
      (blockRows s slot, {projection('s',4)}, {projection('s',5)}, ())
    noncomputable abbrev sharedInnerInputs (s : SharedBlock) (U : Fin 4 → Fin 2 → ExactMatrix Q N 44 1)
        (zc0 zc1 : Fin 85184 → ExactMatrix PQ N 1 1) :=
      {tuple_of(rows + [projection('s',4),projection('s',5)] + [f'U {slot} {port}' for slot in range(4) for port in range(2)] + ['zc0','zc1'])}
    noncomputable abbrev sharedOuterInputs (s : Rows4) (zc0 zc1 : Fin 85184 → ExactMatrix PQ N 1 1)
        (Ar : Fin 4 → Fin 2 → Mat88) :=
      {tuple_of(rows + ['zc0','zc1'] + [f'Ar {slot} {port}' for slot in range(4) for port in range(2)])}

    theorem iterProject {{State Target : Type}} {{body : Nat → State → State → Prop}}
        {{targetBody : Nat → Target → Target → Prop}} (project : State → Target)
        (hstep : ∀ i s t, body i s t → targetBody i (project s) (project t))
        {{count : Nat}} {{s t : State}} (h : MxxIR.IterRuns body count s t) :
        MxxIR.IterRuns targetBody count (project s) (project t) := by
      induction h with
      | zero => exact .zero _
      | step previous run ih => exact .step ih (hstep _ _ _ run)

    set_option maxRecDepth 200000 in
    set_option maxHeartbeats 64000000 in
    theorem sharedInner_project {{model : HashModel}} {{tape : SampleTape}} {{path targetPath : List Nat}}
        {{params : Stage_online_eval.Params}} {{member a j : Nat}} (hm : member < 7)
        {{U : Fin 4 → Fin 2 → ExactMatrix Q N 44 1}}
        {{zc0 zc1 : Fin 85184 → ExactMatrix PQ N 1 1}} {{s t : SharedBlock}}
        (h : Stage_online_eval.sequential_sequential_parallel_generatedRoot_{root}_{outer_id}_48 Backend.backend
          model tape path params member a j (sharedInnerInputs s U zc0 zc1) t) (slot : Fin 4) :
        Stage_online_eval.sequential_sequential_parallel_generatedRoot_58_54_21 Backend.backend model tape
          targetPath params (15 + member) a j
          ((blockProjection s slot).1, (blockProjection s slot).2.1, (blockProjection s slot).2.2.1,
            U slot 0, U slot 1, zc0, zc1, ()) (blockProjection t slot) := by
      have hbranch : 15 + member < 22 := by omega
      have hm0 : 0 ≤ Int.ofNat (15 + member) := by simp only [Int.ofNat_eq_natCast]; omega
      have hm22 : Int.ofNat (15 + member) < 22 := by simp only [Int.ofNat_eq_natCast]; omega
      obtain ⟨w9, w38, w39, w40, w42, w44, w45, w47,
        h0, h1, h2, h3, h4, h5, h6, h7, x9, h8, h9, h10, h11, h12, x38, x39,
        x40, x42, h13, h14, x44, x45, x47, xout⟩ := h
      rw [leafMemberSelect member hm] at h11 h12 x38 h13 h14 x44
      rw [xout]
      fin_cases slot <;>
        refine ⟨w9, w38, w39, w40, w42, w44, w45, w47, ?_⟩ <;>
        dsimp only [Stage_online_eval.sequential_sequential_parallel_generatedRoot_58_54_21.constraints_0]
      all_goals simp only [switchMembers_eq (15 + member) hbranch]
      all_goals exact ⟨h0, h1, h2, h3, h4, h5, h6, h7, x9, h8, hm0, hm22,
        h11, h12, x38, x39, x40, x42, h13, h14, x44, x45, x47, rfl⟩

    SHARED_END
    '''
    outer_scope = f'Stage_online_eval.sequential_parallel_generatedRoot_{root}_{outer_id}'
    facts = (HERE / 'generated/Stage_online_eval_Facts.lean').read_text()
    import re
    chunks = {}
    active = False
    for line in facts.splitlines():
        match = re.match(rf'structure sequential_parallel_generatedRoot_{root}_{outer_id}\.Facts_(\d+) ', line)
        if match:
            active, chunk = True, match[1]
        elif line.startswith(('structure ', 'theorem ')):
            active = False
        if active:
            match = re.match(r'  (h\w+) :', line)
            if match:
                chunks[match[1]] = chunk
    def fact(field):
        return f'F.c{chunks[field]}.{field}'
    def wire(number, port=0):
        return f'{outer_scope}.w_{number}_{port} ctx'
    digits = array_of([array_of([wire(19+8*slot),wire(23+8*slot)]) for slot in range(4)])
    outer = f'''
    set_option maxRecDepth 200000 in
    set_option maxHeartbeats 64000000 in
    theorem sharedOuter_project {{model : HashModel}} {{tape : SampleTape}} {{path targetPath : List Nat}}
        {{params : Stage_online_eval.Params}} {{member a : Nat}} (hm : member < 7)
        {{zc0 zc1 : Fin 85184 → ExactMatrix PQ N 1 1}} {{Ar : Fin 4 → Fin 2 → Mat88}}
        {{s t : Rows4}}
        (h : {outer_scope} Backend.backend model tape path params member a
          (sharedOuterInputs s zc0 zc1 Ar) t) (slot : Fin 4) :
        Stage_online_eval.sequential_parallel_generatedRoot_58_54 Backend.backend model tape targetPath
          params (15 + member) a (rows4 s slot, zc0, zc1, Ar slot 0, Ar slot 1, ()) (rows4 t slot) := by
      have hm0 : 0 ≤ Int.ofNat (15 + member) := by simp only [Int.ofNat_eq_natCast]; omega
      have hm22 : Int.ofNat (15 + member) < 22 := by simp only [Int.ofNat_eq_natCast]; omega
      have h00 : (0 : Nat) ≤ 0 := by omega
      have h01 : (0 : Nat) < 1 := by omega
      have h11 : (1 : Nat) ≤ 1 := by omega
      have ha0 : 0 ≤ 44 + Int.ofNat a := by simp only [Int.ofNat_eq_natCast]; omega
      have ha1 : 44 + Int.ofNat a < 45 + Int.ofNat a := by omega
      have h88 : (0 : Nat) ≤ 88 := by omega
      obtain ⟨w, hb⟩ := h
      let ctx : {outer_scope}.Ctx Backend.backend model tape path params member a :=
        ⟨sharedOuterInputs s zc0 zc1 Ar, w⟩
      have F := {outer_scope}.facts ctx hb
      have ha88 : 45 + Int.ofNat a ≤ 88 := F.c0.t18
      let digits : Fin 4 → Fin 2 → ExactMatrix Q N 44 1 := {digits}
      have hin := {fact('h48_5')}
      change MxxIR.IterRuns (fun j current next ↦
        Stage_online_eval.sequential_sequential_parallel_generatedRoot_{root}_{outer_id}_48 Backend.backend model tape
          (path ++ [48, j]) params member a j (sharedInnerInputs current digits zc0 zc1) next) _ _ _ at hin
      have projected := iterProject (State := SharedBlock) (Target := MulBlock)
        (body := fun j current next ↦
          Stage_online_eval.sequential_sequential_parallel_generatedRoot_{root}_{outer_id}_48 Backend.backend model tape
            (path ++ [48, j]) params member a j (sharedInnerInputs current digits zc0 zc1) next)
        (fun current ↦ blockProjection current slot)
        (targetBody := fun j current next ↦
          Stage_online_eval.sequential_sequential_parallel_generatedRoot_58_54_21 Backend.backend model tape
            (targetPath ++ [21, j]) params (15 + member) a j
            (current.1, current.2.1, current.2.2.1, digits slot 0, digits slot 1, zc0, zc1, ()) next)
        (fun j current next run ↦ sharedInner_project (U := digits) (s := current) (t := next)
          (zc0 := zc0) (zc1 := zc1) hm run slot) hin
      have hmember : MxxIR.selectInt (Int.ofNat (15 + member)) switchMembers =
          MxxIR.selectInt (Int.ofNat member) [15, 16, 17, 18, 19, 20, 21] := by
        rw [switchMembers_eq (15 + member) (by omega), leafMemberSelect member hm]
        simp only [Int.ofNat_eq_natCast]
      have hswitch : MxxIR.selectInt (Int.ofNat (15 + member)) switchList =
          MxxIR.selectInt (Int.ofNat member) leafSwitches := by
        rw [← kSw_cast (15 + member) (by omega), leafSwitchSelect member hm]
        simp only [Int.ofNat_eq_natCast]
      rw [{fact('hout')}]
      fin_cases slot
    '''
    shared = dedent(shared)
    outer = dedent(outer)
    for slot in range(4):
        fields = [(3,0,6,0),(4,0,7,0),(5,0,8,0),(7,0,10,0),(9,0,12,0),(10,0,13,0),(12,0,15,0),
                  (14,0,17+8*slot,0),(15,0,18+8*slot,0),(16,0,19+8*slot,0),
                  (18,0,21+8*slot,0),(19,0,22+8*slot,0),(20,0,23+8*slot,0),
                  (21,0,48,slot),(21,1,48,4),(21,2,48,5)]
        outer += '  · refine ⟨({\n' + ',\n'.join(f'        w_{old}_{port} := {wire(new,newport)}' for old,port,new,newport in fields) + ' } : Stage_online_eval.sequential_parallel_generatedRoot_58_54.Witness), ?_⟩\n'
        outer += '''    dsimp only [Stage_online_eval.sequential_parallel_generatedRoot_58_54.body,
      Stage_online_eval.sequential_parallel_generatedRoot_58_54.constraints_0,
      Stage_online_eval.sequential_parallel_generatedRoot_58_54.constraints_1]
    rw [hmember, hswitch]
    '''
        outer = outer.rstrip() + '\n'
        head = ['hm0', 'hm22']
        head += [fact(field) for field in ['h5_0','h5_0_1','h6_0','h7_0','h8_0','h10_0','h5_0_2','h5_0_3','h12_0','h13_0','h15_0']]
        bounds = ['h00','h01','h11','ha0','ha1','ha88']
        head += bounds + [fact(f'h{17+8*slot}_0'), 'hm0', 'hm22',fact(f'h{18+8*slot}_0'),fact(f'h{19+8*slot}_0')]
        head += bounds + [fact(f'h{21+8*slot}_0'),'hm0']
        tail = ['hm22',fact(f'h{22+8*slot}_0'),fact(f'h{23+8*slot}_0'), 'h88','projected','rfl']
        outer += '    exact ⟨⟨' + ',\n      '.join(head) + '⟩,\n      ' + ',\n      '.join(tail) + '⟩\n'
    shared = shared.replace('SHARED_END\n', outer)

    return shared


def build_shared_loop(group: int) -> str:
    zero_rows = tuple_of(['0'] * 4)
    return r'''
theorem iterOne {State : Type} {body : Nat → State → State → Prop} {s t : State}
    (h : MxxIR.IterRuns body (Int.toNat 1) s t) : body 0 s t := by
  obtain ⟨current, previous, run⟩ := MxxIR.IterRuns.step_of_succ (count := 0) h
  cases previous
  exact run

set_option maxRecDepth 200000 in
set_option maxHeartbeats 8000000 in
theorem sharedLoop {model : HashModel} {tape : SampleTape} {path : List Nat}
    {params : Stage_online_eval.Params} {member : Nat} (hm : member < 7)
    (slot : Fin 16) {A : Fin 2 → Fin 2 → Mat44} {Ar : Fin 16 → Fin 2 → Mat88}
    (hA0 : concatColumns (A 0 0) (A 0 1) (Ar slot 0))
    (hA1 : concatColumns (A 1 0) (A 1 1) (Ar slot 1))
    {zc0 zc1 : Fin 85184 → ExactMatrix PQ N 1 1} {final : Rows16}
    (h : MxxIR.IterRuns (fun a current next ↦
      Stage_online_eval.sequential_parallel_generatedRoot_99_123 Backend.backend model tape
        (path ++ [123, a]) params member a (sharedOuterInputs current zc0 zc1 Ar) next)
      (Int.toNat 44) «zeroRows» final) :
    rows16 final slot = lutCompRow A (expandFam zc0) (expandFam zc1) ⟨15 + member, by omega⟩ := by
  have hb : 15 + member < 22 := by omega
  have initial : rows16 («zeroRows» : Rows16) slot = 0 := by fin_cases slot <;> rfl
  have projected := iterProject (State := Rows16) (Target := Mat88)
    (body := fun a current next ↦
      Stage_online_eval.sequential_parallel_generatedRoot_99_123 Backend.backend model tape
        (path ++ [123, a]) params member a (sharedOuterInputs current zc0 zc1 Ar) next)
    (fun current ↦ rows16 current slot)
    (targetBody := fun a current next ↦
      Stage_online_eval.sequential_parallel_generatedRoot_58_54 Backend.backend model tape
        (path ++ [54, a]) params (15 + member) a
        (current, zc0, zc1, Ar slot 0, Ar slot 1, ()) next)
    (fun a current next run ↦ sharedOuter_project (Ar := Ar) (s := current) (t := next)
      (zc0 := zc0) (zc1 := zc1) hm run slot) h
  dsimp only at projected
  rw [initial] at projected
  apply Matrix.ext
  intro row col
  rw [Subsingleton.elim row 0]
  exact lutLoop56 hA0 hA1 hb projected col
'''.replace('«zeroRows»', zero_rows).replace('Rows16', 'Rows4').replace('rows16', 'rows4').replace(
    'Fin 16', 'Fin 4').replace('generatedRoot_99_123',
    f'generatedRoot_{GROUP_ROOTS[group]}_{OUTER_SCOPES[group]}').replace('[123,', f'[{OUTER_SCOPES[group]},')


def build_member(group: int) -> str:
    root, outer_id = GROUP_ROOTS[group], OUTER_SCOPES[group]
    scopes = SLOT_SCOPES[group]
    start = 4 * group
    facts = (HERE / 'generated/Stage_online_eval_Facts.lean').read_text()
    scope = f'Stage_online_eval.parallel_generatedRoot_{root}'
    chunks = {}
    active = False
    for line in facts.splitlines():
        match = re.match(rf'structure parallel_generatedRoot_{root}\.Facts_(\d+) ', line)
        if match:
            active, chunk = True, match[1]
        elif line.startswith(('structure ', 'theorem ')):
            active = False
        if active:
            match = re.match(r'  (h\w+) :', line)
            if match:
                chunks[match[1]] = chunk

    def fact(field):
        return f'F.c{chunks[field]}.{field}'

    def wire(number, port=0):
        return f'{scope}.w_{number}_{port} ctx'

    def output_state(number):
        return tuple_of([wire(number, port) for port in range(4)])

    inputs = [f'T0 {start} member', f'T1 {start} member', 'first.hf0', 'first.hf1']
    if group == 0:
        inputs += ['first.c0', 'first.c1', 'first.key', 'zc0', 'zc1', 'M0', 'M1',
                   'K00', 'K01', 'K10', 'K11', 'tables', 'T0 1', 'T1 1',
                   'second.c0', 'second.c1', 'T0 2', 'T1 2', 'T0 3', 'T1 3']
    else:
        inputs += ['second.c0', 'second.c1', 'first.key', 'zc0', 'zc1',
                   'K00', 'K01', 'K10', 'K11', 'tables']
        inputs += [f'T{port} {start + slot}' for slot in range(1, 4) for port in range(2)]
    inputs += [f'T0 {start}', f'T1 {start}']
    out = '''
theorem forall4 {pred : Fin 4 → Prop}
'''
    out += ''.join(f'    (h{slot} : pred {slot})\n' for slot in range(4))
    out += '    : ∀ slot, pred slot := by\n  intro slot\n  fin_cases slot\n'
    out += ''.join(f'  · exact h{slot}\n' for slot in range(4))
    terms = [f'term {slot}' for slot in range(4)]
    acc = '0'
    for term in terms:
        acc = f'({acc} + {term})'
    out += f'''
theorem sum4 {{Value : Type}} [AddCommMonoid Value] (term : Fin 4 → Value) :
    ∑ slot, term slot = {acc} := by
  simp only [Fin.sum_univ_succ, Fin.sum_univ_zero]
  abel

noncomputable abbrev memberInputs (first : BranchData) (second : Leaf2Data)
    (T0 T1 : Fin 16 → Fin 7 → Mat88) (member : Fin 7)
    (zc0 zc1 : Fin 85184 → ExactMatrix PQ N 1 1) (M0 M1 : Mat88)
    (K00 K01 K10 K11 : Fin 15 → Mat44) (tables : Fin 512 → Int) :=
  {tuple_of(inputs)}

set_option maxRecDepth 200000 in
set_option maxHeartbeats 8000000 in
theorem memberLane {{model : HashModel}} {{tape : SampleTape}} {{path : List Nat}}
    {{params : Stage_online_eval.Params}} {{member : Nat}} (hm : member < 7)
    (first : BranchData) (second : Leaf2Data)
    (hf0 : second.hf0 = first.hf0) (hf1 : second.hf1 = first.hf1) (hkey : second.key = first.key)
    {{T0 T1 : Fin 16 → Fin 7 → Mat88}}
    (hT0 : ∀ slot, (leafSlotData first second slot).TL0 = T0 slot)
    (hT1 : ∀ slot, (leafSlotData first second slot).TL1 = T1 slot)
    {{M : Fin 2 → Fin 2 → Mat44}} {{M0 M1 : Mat88}}
    (hM0 : concatColumns (M 0 0) (M 0 1) M0) (hM1 : concatColumns (M 1 0) (M 1 1) M1)
    {{K00 K01 K10 K11 : Fin 15 → Mat44}}
    {{zc0 zc1 : Fin 85184 → ExactMatrix PQ N 1 1}} {{tables : Fin 512 → Int}}
    (hE : ∀ slot : Fin 16, (leafSlotData first second slot).Eb (15 + member) (by omega) =
      lutCompRow (Fin.cases M (parentKeys K00 K01 K10 K11) slot)
        (expandFam zc0) (expandFam zc1) ⟨15 + member, by omega⟩)
    {{final : Mat44 × Mat44 × Unit}}
    (h : {scope} Backend.backend model tape path params member
      (memberInputs first second T0 T1 ⟨member, hm⟩ zc0 zc1 M0 M1 K00 K01 K10 K11 tables) final) :
    (∀ slot : Fin 4, (leafSlotData first second (chunkSlot {group} slot)).Runs model
      (15 + member) (by omega)) ∧
      (∀ port : Fin 2, ![final.1, final.2.1] port =
        ∑ slot : Fin 4, memberTerm model first second
          (fun index ↦ branchCoef tables 0 index (by norm_num)) ⟨member, hm⟩
          (chunkSlot {group} slot) port) := by
  have hb : 15 + member < 22 := by omega
'''
    parents = range(max(0, start - 1), start + 3)
    for parent in parents:
        out += f'  have select{parent} := slotParentSelect (⟨{parent}, by decide⟩ : Fin 15) member hm\n'
    out += f'''  obtain ⟨w, body⟩ := h
  let ctx : {scope}.Ctx Backend.backend model tape path params member :=
    ⟨memberInputs first second T0 T1 ⟨member, hm⟩ zc0 zc1 M0 M1 K00 K01 K10 K11 tables, w⟩
  have F := {scope}.facts ctx body
'''

    def read_parent(read, parent):
        return f'''  have read{read} := {fact(f'h{read}_0')}
  change familyGetDynamic _ (MxxIR.selectInt (MxxIR.selectInt (Int.ofNat member)
    (List.ofFn fun lane : Fin 7 ↦ Int.ofNat (15 * lane + {parent}))) parentList2) _ at read{read}
  rw [select{parent}] at read{read}
  have eq{read} := (familyGet_eq read{read}).2
'''

    joined = []
    concats = []
    for local in range(4):
        slot = start + local
        if slot == 0:
            joined.append(['M0', 'M1'])
            concats.append(['hM0', 'hM1'])
            continue
        ordinal = local - 1 if group == 0 else local
        selector = 15 if ordinal == 0 and group == 0 else (7 if ordinal == 0 else 25 + 7 * (ordinal - 1))
        reads = [16, 18, 21, 23] if ordinal == 0 else [selector + offset for offset in [1, 2, 4, 5]]
        joins = [19, 24] if ordinal == 0 else [selector + 3, selector + 6]
        for read in reads:
            out += read_parent(read, slot - 1)
        for port, join in enumerate(joins):
            out += f'''  have join{join} := {fact(f'h{join}_0')}
  rw [eq{reads[2 * port]}, eq{reads[2 * port + 1]}] at join{join}
'''
        joined.append([wire(join) for join in joins])
        concats.append([f'join{join}' for join in joins])
    out += f'  let joined : Fin 4 → Fin 2 → Mat88 := {array_of([array_of(pair) for pair in joined])}\n'
    out += f'''  have companions := {fact(f'h{outer_id}_3')}
  change MxxIR.IterRuns (fun a current next ↦
    Stage_online_eval.sequential_parallel_generatedRoot_{root}_{outer_id} Backend.backend model tape
      (path ++ [{outer_id}, a]) params member a (sharedOuterInputs current zc0 zc1 joined) next) _ _ _ at companions
'''
    for local in range(4):
        slot = start + local
        key = 'M' if slot == 0 else f'parentKeys K00 K01 K10 K11 (⟨{slot - 1}, by decide⟩ : Fin 15)'
        out += f'''  have companion{slot} : {wire(outer_id, local)} =
      (leafSlotData first second {slot}).Eb (15 + member) hb := by
    apply Eq.trans ?_ (hE {slot}).symm
    exact sharedLoop hm {local} (A := {key}) (Ar := joined)
      (zc0 := zc0) (zc1 := zc1) {' '.join(concats[local])} companions
'''
    for local, slot_scope in enumerate(scopes):
        slot = start + local
        if local == 0:
            initial = tuple_of(['0', '0', f'T0 {slot} ⟨member, hm⟩', f'T1 {slot} ⟨member, hm⟩'])
            hT = f'⟨congrFun (hT0 {slot}).symm ⟨member, hm⟩, congrFun (hT1 {slot}).symm ⟨member, hm⟩⟩'
        else:
            initial = output_state(scopes[local - 1])
            hT = f'⟨(slot{slot - 1}.2.2.2.1).trans (congrFun (hT0 {slot}).symm ⟨member, hm⟩), ' \
                 f'(slot{slot - 1}.2.2.2.2).trans (congrFun (hT1 {slot}).symm ⟨member, hm⟩)⟩'
        d = 'first' if slot == 0 else f'second.memberBranch (⟨{slot - 1}, by decide⟩ : Fin 15)'
        reads = None
        if slot != 0:
            if local == 0:
                reads = [8, 10]
            elif group == 0 and local == 1:
                reads = [45, 47]
            else:
                reads = [scopes[local - 1] + 1, scopes[local - 1] + 2]
            for read in reads:
                out += read_parent(read, slot - 1)
        out += f'  have run{slot} := iterOne {fact(f"h{slot_scope}_3")}\n'
        if reads:
            out += f'  rw [eq{reads[0]}, eq{reads[1]}] at run{slot}\n'
        out += f'''  have slot{slot} := slotFinish{slot} hm ({d}) (s := {initial})
    (t := {output_state(slot_scope)}) companion{slot} {hT}
    (by simpa only [Leaf2Data.memberBranch, hf0, hf1, hkey] using run{slot})
'''
    out += '  dsimp only at ' + ' '.join(f'slot{start + local}' for local in range(4)) + '\n'
    out += f'  refine ⟨forall4 (pred := fun slot ↦ (leafSlotData first second (chunkSlot {group} slot)).Runs model (15 + member) hb) ' + ' '.join(f'slot{start + local}.1' for local in range(4)) + ', ?_⟩\n'
    out += '  intro port\n  fin_cases port\n'
    for port in range(2):
        out += f'  · rw [{fact("hout")}]\n'
        out += f'    change {wire(scopes[-1], port)} = _\n'
        out += '    rw [sum4]\n'
        out += '    change _ = 0' + ''.join(f' + memberTerm model first second (fun index ↦ branchCoef tables 0 index (by norm_num)) ⟨member, hm⟩ {start + local} {port}' for local in range(4)) + '\n'
        out += '    dsimp only [memberTerm, leafSlotData, Fin.cases_zero, Fin.cases_succ]\n'
        projection_path = '.2.1' if port == 0 else '.2.2.1'
        out += '    rw [' + ', '.join(f'slot{start + local}{projection_path}' for local in reversed(range(4))) + ']\n'
        out += '    rfl\n'
    if group:
        out = out.replace('(M0 M1 : Mat88)', '(_M0 _M1 : Mat88)')
        out = out.replace('(hM0 :', '(_hM0 :').replace('(hM1 :', '(_hM1 :')
    return out


def main() -> None:
    source = (HERE / "PeOnlineStep0.lean").read_text()
    source = source.split("/-! ## Member-group slot bodies -/")[0]
    source = source.removesuffix("end PeProof.Step0\n")
    def render_slot(group: int, slot: int, scope: int) -> str:
        members = list(range(15, 22)) if slot == 0 else [15 * member + slot - 1 for member in range(7)]
        select = "MxxIR.selectInt (MxxIR.selectInt (Int.ofNat b) [" + ", ".join(map(str, members))
        select += "]) " + ("switchList" if slot == 0 else "coefList2")
        return (SLOT_TEMPLATE.replace("«slot»", str(slot)).replace("«scope»", str(scope))
                .replace("«coefSelect»", select).replace('generatedRoot_99_',
                    f'generatedRoot_{GROUP_ROOTS[group]}_'))
    sums = source[source.index('set_option maxHeartbeats 4000000 in\n/-- One iteration `i` of the sum'):]
    sums = sums[:sums.index('/-! ## The first-depth leaves')]
    sums = sums.replace('sumStep63', 'memberSumStep').replace('sumLoop63', 'memberSumLoop')
    sums = re.sub(r'\b15\b', '7', sums)
    sums = re.sub(r'\b13\b', '5', sums)
    sums = sums.replace('internal branches', 'member lanes').replace('other thirteen', 'other five')
    source += "/-! ## Member-group slot bodies -/\n"
    for group in range(4):
        slots = "\n".join(render_slot(group, 4 * group + slot, scope)
                          for slot, scope in enumerate(SLOT_SCOPES[group]))
        group_sums = sums.replace('_65', f'_{GROUP_ROOTS[group] + 7}').replace(
            '[65,', f'[{GROUP_ROOTS[group] + 7},')
        source += (f'\nnamespace Group{group}\n' + slots + group_sums + build_shared(group)
                   + build_shared_loop(group) + build_member(group) + f'\nend Group{group}\n')
    source += "\nend PeProof.Step0\n"
    (HERE / "PeOnlineStep0.lean").write_text(source)
    for step in (1, 2, 3):
        (HERE / f"PeOnlineStep{step}.lean").write_text(derive(source, step))



if __name__ == "__main__":
    main()
