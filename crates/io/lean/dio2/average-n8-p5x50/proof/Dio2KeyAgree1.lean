import Dio2Preimage
import Dio2Online1State
import Dio2KeysLoops
import Gen_W_Stage_keys_online_1_1_Scopes
import W_Stage_keys_online_1_1_Facts

/-!
# Key agreement for the last online step

The last preimage solves the sampled anchor equation against the keys stage's
`dio2/output-key`. The state and smudge keys depend only on the common hash key,
switching families, helper key tags, RHS artifacts and gather tables. Ciphertext
rows and label payloads are deliberately not compared: the role-key lemmas
compare each graph bundle's key projection, quantified over all members.

`Link1` records only actual common sources in `W_Claim`; `claim_link1` verifies
that wiring. `online1_hpre` and `claim_hpreS1` discharge the exact preimage
hypothesis of `online1_output`, with anchor rows as `Bk`.
-/

set_option maxHeartbeats 400000000
set_option maxRecDepth 16384

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof
  RefreshTopProof.Gen RefreshTopProof.GenInst RefreshTopProof.GenFinal RefreshTopProof.GenIface

variable {n : Nat}

/-- **Two stacked products agree on their key rows.** Rows `1` and `2` of `[c; A₀; A₁] d` depend
on the key rows `A` and `d` alone, not on the ciphertext row `c`. -/
theorem key1_stack_rows_agree {q m m' : Nat} {l0 l1 l2 r0 r1 r2 l0' l1' l2' r0' r1' r2' :
  ExactMatrix q n 1 m}
    {l01 r01 l01' r01' : ExactMatrix q n (1 + 1) m} {l r l' r' : ExactMatrix q n (1 + 1 + 1) m}
    {both both' : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows l0 l1 l01) (hl : concatRows l01 l2 l)
    (hr01 : concatRows r0 r1 r01) (hr : concatRows r01 r2 r) (hboth : concatColumns l r both)
    (hl01' : concatRows l0' l1' l01') (hl' : concatRows l01' l2' l')
    (hr01' : concatRows r0' r1' r01') (hr' : concatRows r01' r2' r') (hboth' : concatColumns l' r'
      both')
    (e1 : l1 = l1') (e2 : l2 = l2') (e3 : r1 = r1') (e4 : r2 = r2') {d d' : ExactMatrix q n (m + m)
      m'}
    (ed : d = d') :
    MxxExponentLut.rowOf (both * d) 1 = MxxExponentLut.rowOf (both' * d') 1 ∧
      MxxExponentLut.rowOf (both * d) 2 = MxxExponentLut.rowOf (both' * d') 2 := by
  obtain ⟨-, a1, a2⟩ := stacked_row hl01 hl hr01 hr hboth d
  obtain ⟨-, b1, b2⟩ := stacked_row hl01' hl' hr01' hr' hboth' d'
  rw [a1, a2, b1, b2, e1, e2, e3, e4, ed]
  exact ⟨rfl, rfl⟩

/-- The inputs the online stage of step 1 and its keys stage read from the same source (`W_Claim`).
  -/
structure Link1 {backend : BackendContext} {hashModel : HashModel} {params :
  W_Stage_online_1.Params}
    {paramsK : W_Stage_keys_online_1_1.Params}
    (Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params)
    (Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK) : Prop where
  e5 : W_Stage_online_1.generatedRoot.w_5_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_1_0 Ck
  e7 : W_Stage_online_1.generatedRoot.w_7_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_3_0 Ck
  e9 : W_Stage_online_1.generatedRoot.w_9_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_5_0 Ck
  e11 : W_Stage_online_1.generatedRoot.w_11_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_7_0 Ck
  e14 : W_Stage_online_1.generatedRoot.w_14_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_10_0 Ck
  e70 : W_Stage_online_1.generatedRoot.w_70_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_66_0 Ck
  e72 : W_Stage_online_1.generatedRoot.w_72_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_68_0 Ck
  e82 : W_Stage_online_1.generatedRoot.w_82_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_78_0 Ck
  e132 : W_Stage_online_1.generatedRoot.w_132_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_126_0
    Ck
  e133 : W_Stage_online_1.generatedRoot.w_133_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_127_0
    Ck
  e134 : W_Stage_online_1.generatedRoot.w_134_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_128_0
    Ck
  e135 : W_Stage_online_1.generatedRoot.w_135_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_129_0
    Ck
  e138 : W_Stage_online_1.generatedRoot.w_138_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_132_0
    Ck
  e139 : W_Stage_online_1.generatedRoot.w_139_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_133_0
    Ck
  e146 : W_Stage_online_1.generatedRoot.w_146_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_140_0
    Ck
  e147 : W_Stage_online_1.generatedRoot.w_147_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_141_0
    Ck
  e150 : W_Stage_online_1.generatedRoot.w_150_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_144_0
    Ck
  e151 : W_Stage_online_1.generatedRoot.w_151_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_145_0
    Ck

set_option maxHeartbeats 400000000 in
/-- **The root LUT loop `sequential_generatedRoot_91` of step 1 has the same keys and carried
  switching ciphertexts** as the keys stage's `sequential_generatedRoot_87` (`iter_link_gen` with
  the step forms of `LutRoundStep.keys_1`). -/
theorem state_loop_agree1 {backend : BackendContext} {hashModel : HashModel} {params :
  W_Stage_online_1.Params}
    {paramsK : W_Stage_keys_online_1_1.Params}
    {Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params} {oon : _}
    (Fon : W_Stage_online_1.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_1_1.generatedRoot.Facts Ck ok) (L : Link1 Con Ck)
    (hq1 : 1 < W_Stage_online_1.modulus_0) (layout : RegularLayout W_Stage_online_1.modulus_0)
    (hlayout : backend.regularLayout W_Stage_online_1.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    W_Stage_online_1.generatedRoot.w_91_2 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_2 Ck ∧
      W_Stage_online_1.generatedRoot.w_91_3 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_3 Ck ∧
      W_Stage_online_1.generatedRoot.w_91_4 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_4 Ck ∧
      W_Stage_online_1.generatedRoot.w_91_5 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_5 Ck ∧
      W_Stage_online_1.generatedRoot.w_91_6 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_6 Ck ∧
      W_Stage_online_1.generatedRoot.w_91_7 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_7 Ck ∧
      W_Stage_online_1.generatedRoot.w_91_8 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_8 Ck ∧
      W_Stage_online_1.generatedRoot.w_91_9 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_9 Ck :=
      by
  have ha3 := Fon.c0.h20_0
  rw [L.e14] at ha3
  have a3 : (W_Stage_online_1.generatedRoot.w_20_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_16_0 Ck) := (MxxExponentLut.hashSample_eq ha3).trans
    (MxxExponentLut.hashSample_eq Fk.c0.h16_0).symm
  have a4 : (W_Stage_online_1.generatedRoot.w_21_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_17_0 Ck) := rfl
  have a2 : (W_Stage_online_1.generatedRoot.w_22_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_18_0 Ck) := by
    unfold W_Stage_online_1.generatedRoot.w_22_0 W_Stage_keys_online_1_1.generatedRoot.w_18_0
    rw [a3, a4]
  have ha6 := Fon.c0.h23_0
  rw [L.e14] at ha6
  have a6 : (W_Stage_online_1.generatedRoot.w_23_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_19_0 Ck) := (MxxExponentLut.hashSample_eq ha6).trans
    (MxxExponentLut.hashSample_eq Fk.c0.h19_0).symm
  have a7 : (W_Stage_online_1.generatedRoot.w_24_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_20_0 Ck) := rfl
  have a5 : (W_Stage_online_1.generatedRoot.w_25_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_21_0 Ck) := by
    unfold W_Stage_online_1.generatedRoot.w_25_0 W_Stage_keys_online_1_1.generatedRoot.w_21_0
    rw [a6, a7]
  have ha9 := Fon.c0.h31_0
  rw [L.e14] at ha9
  have a9 : (W_Stage_online_1.generatedRoot.w_31_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_27_0 Ck) := (MxxExponentLut.hashSample_eq ha9).trans
    (MxxExponentLut.hashSample_eq Fk.c0.h27_0).symm
  have a10 : (W_Stage_online_1.generatedRoot.w_32_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_28_0 Ck) := rfl
  have a8 : (W_Stage_online_1.generatedRoot.w_33_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_29_0 Ck) := by
    unfold W_Stage_online_1.generatedRoot.w_33_0 W_Stage_keys_online_1_1.generatedRoot.w_29_0
    rw [a9, a10]
  have ha12 := Fon.c0.h34_0
  rw [L.e14] at ha12
  have a12 : (W_Stage_online_1.generatedRoot.w_34_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_30_0 Ck) := (MxxExponentLut.hashSample_eq ha12).trans
    (MxxExponentLut.hashSample_eq Fk.c0.h30_0).symm
  have a13 : (W_Stage_online_1.generatedRoot.w_35_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_31_0 Ck) := rfl
  have a11 : (W_Stage_online_1.generatedRoot.w_36_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_32_0 Ck) := by
    unfold W_Stage_online_1.generatedRoot.w_36_0 W_Stage_keys_online_1_1.generatedRoot.w_32_0
    rw [a12, a13]
  have ha17 := Fon.c0.h39_0
  rw [L.e14] at ha17
  have a17 : (W_Stage_online_1.generatedRoot.w_39_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_35_0 Ck) := (MxxExponentLut.hashSample_eq ha17).trans
    (MxxExponentLut.hashSample_eq Fk.c0.h35_0).symm
  have ha18 := Fon.c0.h40_0
  rw [L.e14] at ha18
  have a18 : (W_Stage_online_1.generatedRoot.w_40_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_36_0 Ck) := (MxxExponentLut.hashSample_eq ha18).trans
    (MxxExponentLut.hashSample_eq Fk.c0.h36_0).symm
  have ha16 := Fon.c0.h41_0
  rw [a17, a18] at ha16
  have a16 : (W_Stage_online_1.generatedRoot.w_41_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_37_0 Ck) := concatRows_unique ha16 Fk.c0.h37_0
  have ha15 := Fon.c0.h42_0
  rw [a16] at ha15
  have a15 : (W_Stage_online_1.generatedRoot.w_42_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_38_0 Ck) := gadgetDecomposeRuns_unique ha15 Fk.c0.h38_0
  have ha14 := Fon.c0.h43_0
  rw [a15] at ha14
  have a14 : (W_Stage_online_1.generatedRoot.w_43_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_39_0 Ck) := compactCenteredRebaseRuns_unique ha14
    Fk.c0.h39_0
  have a1 : (W_Stage_online_1.generatedRoot.w_53_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_49_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c1.h53_0, slice_rowOfG (by norm_num) Fk.c1.h49_0]
    exact (key1_stack_rows_agree Fon.c0.h22_0 Fon.c0.h26_0 Fon.c0.h33_0 Fon.c0.h37_0 Fon.c0.h38_0
      Fk.c0.h18_0 Fk.c0.h22_0 Fk.c0.h29_0 Fk.c0.h33_0 Fk.c0.h34_0 a2 a5 a8 a11 a14).1
  have a19 : (W_Stage_online_1.generatedRoot.w_54_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_50_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c1.h54_0, slice_rowOfG (by norm_num) Fk.c1.h50_0]
    exact (key1_stack_rows_agree Fon.c0.h22_0 Fon.c0.h26_0 Fon.c0.h33_0 Fon.c0.h37_0 Fon.c0.h38_0
      Fk.c0.h18_0 Fk.c0.h22_0 Fk.c0.h29_0 Fk.c0.h33_0 Fk.c0.h34_0 a2 a5 a8 a11 a14).2
  have ha24 := Fon.c1.h56_0
  rw [L.e14] at ha24
  have a24 : (W_Stage_online_1.generatedRoot.w_56_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_52_0 Ck) := (MxxExponentLut.hashSample_eq ha24).trans
    (MxxExponentLut.hashSample_eq Fk.c1.h52_0).symm
  have ha25 := Fon.c1.h57_0
  rw [L.e14] at ha25
  have a25 : (W_Stage_online_1.generatedRoot.w_57_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_53_0 Ck) := (MxxExponentLut.hashSample_eq ha25).trans
    (MxxExponentLut.hashSample_eq Fk.c1.h53_0).symm
  have ha23 := Fon.c1.h58_0
  rw [a24, a25] at ha23
  have a23 : (W_Stage_online_1.generatedRoot.w_58_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_54_0 Ck) := concatRows_unique ha23 Fk.c1.h54_0
  have ha22 := Fon.c1.h59_0
  rw [a23] at ha22
  have a22 : (W_Stage_online_1.generatedRoot.w_59_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_55_0 Ck) := gadgetDecomposeRuns_unique ha22 Fk.c1.h55_0
  have ha21 := Fon.c1.h60_0
  rw [a22] at ha21
  have a21 : (W_Stage_online_1.generatedRoot.w_60_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_56_0 Ck) := compactCenteredRebaseRuns_unique ha21
    Fk.c1.h56_0
  have a20 : (W_Stage_online_1.generatedRoot.w_66_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_62_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c1.h66_0, slice_rowOfG (by norm_num) Fk.c1.h62_0]
    exact (key1_stack_rows_agree Fon.c0.h22_0 Fon.c0.h26_0 Fon.c0.h33_0 Fon.c0.h37_0 Fon.c0.h38_0
      Fk.c0.h18_0 Fk.c0.h22_0 Fk.c0.h29_0 Fk.c0.h33_0 Fk.c0.h34_0 a2 a5 a8 a11 a21).1
  have a26 : (W_Stage_online_1.generatedRoot.w_67_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_63_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c2.h67_0, slice_rowOfG (by norm_num) Fk.c2.h63_0]
    exact (key1_stack_rows_agree Fon.c0.h22_0 Fon.c0.h26_0 Fon.c0.h33_0 Fon.c0.h37_0 Fon.c0.h38_0
      Fk.c0.h18_0 Fk.c0.h22_0 Fk.c0.h29_0 Fk.c0.h33_0 Fk.c0.h34_0 a2 a5 a8 a11 a21).2
  have ha30 := Fon.c2.h71_0
  rw [L.e70] at ha30
  have a30 : (W_Stage_online_1.generatedRoot.w_71_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_67_0 Ck) := (MxxExponentLut.hashSample_eq ha30).trans
    (MxxExponentLut.hashSample_eq Fk.c2.h67_0).symm
  have ha31 := Fon.c2.h73_0
  rw [L.e72] at ha31
  have a31 : (W_Stage_online_1.generatedRoot.w_73_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_69_0 Ck) := familyGetDynamic_unique ha31 Fk.c2.h69_0
  have ha29 := Fon.c2.h74_0
  rw [a30, a31] at ha29
  have a29 : (W_Stage_online_1.generatedRoot.w_74_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_70_0 Ck) := concatRows_unique ha29 Fk.c2.h70_0
  have ha28 := Fon.c2.h75_0
  rw [a29] at ha28
  have a28 : (W_Stage_online_1.generatedRoot.w_75_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_71_0 Ck) := gadgetDecomposeRuns_unique ha28 Fk.c2.h71_0
  have ha27 := Fon.c2.h76_0
  rw [a28] at ha27
  have a27 : (W_Stage_online_1.generatedRoot.w_76_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_72_0 Ck) := compactCenteredRebaseRuns_unique ha27
    Fk.c2.h72_0
  have a0 : (W_Stage_online_1.generatedRoot.w_79_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_75_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c2.h79_0, slice_rowOfG (by norm_num) Fk.c2.h75_0]
    exact (key1_stack_rows_agree Fon.c1.h53_0_1 Fon.c1.h55_0 Fon.c2.h66_0_1 Fon.c2.h68_0
      Fon.c2.h69_0 Fk.c1.h49_0_1 Fk.c1.h51_0 Fk.c2.h62_0_1 Fk.c2.h64_0 Fk.c2.h65_0 a1 a19 a20 a26
      a27).1
  have a32 : (W_Stage_online_1.generatedRoot.w_80_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_76_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c3.h80_0, slice_rowOfG (by norm_num) Fk.c3.h76_0]
    exact (key1_stack_rows_agree Fon.c1.h53_0_1 Fon.c1.h55_0 Fon.c2.h66_0_1 Fon.c2.h68_0
      Fon.c2.h69_0 Fk.c1.h49_0_1 Fk.c1.h51_0 Fk.c2.h62_0_1 Fk.c2.h64_0 Fk.c2.h65_0 a1 a19 a20 a26
      a27).2
  have ha37 := Fon.c3.h81_0
  rw [L.e70] at ha37
  have a37 : (W_Stage_online_1.generatedRoot.w_81_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_77_0 Ck) := (MxxExponentLut.hashSample_eq ha37).trans
    (MxxExponentLut.hashSample_eq Fk.c3.h77_0).symm
  have ha38 := Fon.c3.h83_0
  rw [L.e82] at ha38
  have a38 : (W_Stage_online_1.generatedRoot.w_83_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_79_0 Ck) := familyGetDynamic_unique ha38 Fk.c3.h79_0
  have ha36 := Fon.c3.h84_0
  rw [a37, a38] at ha36
  have a36 : (W_Stage_online_1.generatedRoot.w_84_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_80_0 Ck) := concatRows_unique ha36 Fk.c3.h80_0
  have ha35 := Fon.c3.h85_0
  rw [a36] at ha35
  have a35 : (W_Stage_online_1.generatedRoot.w_85_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_81_0 Ck) := gadgetDecomposeRuns_unique ha35 Fk.c3.h81_0
  have ha34 := Fon.c3.h86_0
  rw [a35] at ha34
  have a34 : (W_Stage_online_1.generatedRoot.w_86_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_82_0 Ck) := compactCenteredRebaseRuns_unique ha34
    Fk.c3.h82_0
  have a33 : (W_Stage_online_1.generatedRoot.w_89_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_85_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c3.h89_0, slice_rowOfG (by norm_num) Fk.c3.h85_0]
    exact (key1_stack_rows_agree Fon.c1.h53_0_1 Fon.c1.h55_0 Fon.c2.h66_0_1 Fon.c2.h68_0
      Fon.c2.h69_0 Fk.c1.h49_0_1 Fk.c1.h51_0 Fk.c2.h62_0_1 Fk.c2.h64_0 Fk.c2.h65_0 a1 a19 a20 a26
      a34).1
  have a39 : (W_Stage_online_1.generatedRoot.w_90_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_86_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c3.h90_0, slice_rowOfG (by norm_num) Fk.c3.h86_0]
    exact (key1_stack_rows_agree Fon.c1.h53_0_1 Fon.c1.h55_0 Fon.c2.h66_0_1 Fon.c2.h68_0
      Fon.c2.h69_0 Fk.c1.h49_0_1 Fk.c1.h51_0 Fk.c2.h62_0_1 Fk.c2.h64_0 Fk.c2.h65_0 a1 a19 a20 a26
      a34).2
  have a40 : (W_Stage_online_1.generatedRoot.w_4_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_0_0 Ck) := rfl
  have ha41 := Fon.c0.h6_0
  rw [L.e5] at ha41
  have a41 : (W_Stage_online_1.generatedRoot.w_6_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_2_0 Ck) := familyGetDynamic_unique ha41 Fk.c0.h2_0
  have ha42 := Fon.c0.h8_0
  rw [L.e7] at ha42
  have a42 : (W_Stage_online_1.generatedRoot.w_8_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_4_0 Ck) := familyGetDynamic_unique ha42 Fk.c0.h4_0
  have ha43 := Fon.c0.h10_0
  rw [L.e9] at ha43
  have a43 : (W_Stage_online_1.generatedRoot.w_10_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_6_0 Ck) := familyGetDynamic_unique ha43 Fk.c0.h6_0
  have ha44 := Fon.c0.h12_0
  rw [L.e11] at ha44
  have a44 : (W_Stage_online_1.generatedRoot.w_12_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_8_0 Ck) := familyGetDynamic_unique ha44 Fk.c0.h8_0
  let p : (ExactMatrix W_Stage_online_1.modulus_0 8 1 25 × ExactMatrix W_Stage_online_1.modulus_0 8
    1 25 × ExactMatrix W_Stage_online_1.modulus_0 8 1 25 × ExactMatrix W_Stage_online_1.modulus_0 8
    1 25 × ExactMatrix W_Stage_online_1.modulus_0 8 1 25 × ExactMatrix W_Stage_online_1.modulus_0 8
    1 25 × ExactMatrix W_Stage_online_1.modulus_0 8 1 25 × ExactMatrix W_Stage_online_1.modulus_0 8
    1 25 × ExactMatrix W_Stage_online_1.modulus_0 8 1 25 × ExactMatrix W_Stage_online_1.modulus_0 8
    1 25 × Unit) → (Fin 2 → Fin 2 → ExactMatrix W_Stage_online_1.modulus_0 8 1 25) × (Fin 2 → Fin 2
    → ExactMatrix W_Stage_online_1.modulus_0 8 1 25) := fun s ↦
    (![![s.2.2.1, s.2.2.2.1], ![s.2.2.2.2.1, s.2.2.2.2.2.1]],
      ![![s.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.1], ![s.2.2.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.2.2.1]])
  let fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix W_Stage_online_1.modulus_0 8 1 25 :=
    ![![W_Stage_online_1.generatedRoot.w_5_0 Con, W_Stage_online_1.generatedRoot.w_7_0 Con],
    ![W_Stage_online_1.generatedRoot.w_9_0 Con, W_Stage_online_1.generatedRoot.w_11_0 Con]]
  let famN : Nat → Fin 2 → Fin 2 → ExactMatrix W_Stage_online_1.modulus_0 8 1 25 := fun j k l ↦ if h
    : j < 4 then fam k l ⟨j, h⟩ else 0
  let Ak : Fin 2 → Fin 2 → ExactMatrix W_Stage_online_1.modulus_0 8 1 25 :=
    ![![W_Stage_online_1.generatedRoot.w_79_0 Con, W_Stage_online_1.generatedRoot.w_89_0 Con],
    ![W_Stage_online_1.generatedRoot.w_80_0 Con, W_Stage_online_1.generatedRoot.w_90_0 Con]]
  let F : Nat → (Fin 2 → Fin 2 → ExactMatrix W_Stage_online_1.modulus_0 8 1 25) × (Fin 2 → Fin 2 →
    ExactMatrix W_Stage_online_1.modulus_0 8 1 25) → (Fin 2 → Fin 2 → ExactMatrix
    W_Stage_online_1.modulus_0 8 1 25) × (Fin 2 → Fin 2 → ExactMatrix W_Stage_online_1.modulus_0 8 1
    25) :=
    fun i x ↦ (x.1 + branchKeyF backend 4096 25 (MxxIR.selectInt (Int.ofNat i) [1, 3, 5, 15]).toNat
      Ak
      (keyPairG hashModel (W_Stage_online_1.generatedRoot.w_14_0 Con) W_Stage_online_1.modulus_0 25
        helperPre [.integer (MxxIR.selectInt (Int.ofNat i) [1, 3, 5, 15])]) x.2, famN (i + 1))
  have h0 : p (W_Stage_online_1.generatedRoot.w_4_0 Con, (W_Stage_online_1.generatedRoot.w_4_0 Con,
    (W_Stage_online_1.generatedRoot.w_4_0 Con, (W_Stage_online_1.generatedRoot.w_4_0 Con,
    (W_Stage_online_1.generatedRoot.w_4_0 Con, (W_Stage_online_1.generatedRoot.w_4_0 Con,
    (W_Stage_online_1.generatedRoot.w_6_0 Con, (W_Stage_online_1.generatedRoot.w_8_0 Con,
    (W_Stage_online_1.generatedRoot.w_10_0 Con, (W_Stage_online_1.generatedRoot.w_12_0 Con,
    ())))))))))) = p (W_Stage_keys_online_1_1.generatedRoot.w_0_0 Ck,
    (W_Stage_keys_online_1_1.generatedRoot.w_0_0 Ck, (W_Stage_keys_online_1_1.generatedRoot.w_0_0
    Ck, (W_Stage_keys_online_1_1.generatedRoot.w_0_0 Ck,
    (W_Stage_keys_online_1_1.generatedRoot.w_0_0 Ck, (W_Stage_keys_online_1_1.generatedRoot.w_0_0
    Ck, (W_Stage_keys_online_1_1.generatedRoot.w_2_0 Ck,
    (W_Stage_keys_online_1_1.generatedRoot.w_4_0 Ck, (W_Stage_keys_online_1_1.generatedRoot.w_6_0
    Ck, (W_Stage_keys_online_1_1.generatedRoot.w_8_0 Ck, ())))))))))) := by
    rw [a40, a41, a42, a43, a44]
  have hl := iter_link_gen p p F (count := 3)
    (fun i cur nxt hi hb ↦ by
      obtain ⟨hk, hsw⟩ := LutRoundStep.keys_1 (by norm_num) hq1 layout hlayout hexact
        (W_Stage_online_1.sequential_generatedRoot_91.sem hb)
      obtain ⟨n0, n1, n2, n3⟩ := hsw (by omega)
      refine Prod.ext hk ?_
      funext k l
      fin_cases k <;> fin_cases l
      · show _ = famN (i + 1) 0 0; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n0
      · show _ = famN (i + 1) 0 1; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n1
      · show _ = famN (i + 1) 1 0; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n2
      · show _ = famN (i + 1) 1 1; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n3)
    (fun i cur nxt hi hb ↦ by
      obtain ⟨hk, hsw⟩ := LutRoundStep.keys_1 (by norm_num) hq1 layout hlayout hexact
        (W_Stage_keys_online_1_1.sequential_generatedRoot_87.sem hb)
      obtain ⟨n0, n1, n2, n3⟩ := hsw (by omega)
      rw [← a0, ← a32, ← a33, ← a39, ← L.e14] at hk
      rw [← L.e5] at n0
      rw [← L.e7] at n1
      rw [← L.e9] at n2
      rw [← L.e11] at n3
      refine Prod.ext hk ?_
      funext k l
      fin_cases k <;> fin_cases l
      · show _ = famN (i + 1) 0 0; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n0
      · show _ = famN (i + 1) 0 1; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n1
      · show _ = famN (i + 1) 1 0; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n2
      · show _ = famN (i + 1) 1 1; simp only [famN, dif_pos (show i + 1 < 4 by omega)]; exact n3)
    Fon.c3.h91_9 Fk.c3.h87_9 h0
  have t2 : W_Stage_online_1.generatedRoot.w_91_2 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_2
    Ck := congrFun (congrFun (congrArg Prod.fst hl) 0) 0
  have t3 : W_Stage_online_1.generatedRoot.w_91_3 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_3
    Ck := congrFun (congrFun (congrArg Prod.fst hl) 0) 1
  have t4 : W_Stage_online_1.generatedRoot.w_91_4 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_4
    Ck := congrFun (congrFun (congrArg Prod.fst hl) 1) 0
  have t5 : W_Stage_online_1.generatedRoot.w_91_5 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_5
    Ck := congrFun (congrFun (congrArg Prod.fst hl) 1) 1
  have t6 : W_Stage_online_1.generatedRoot.w_91_6 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_6
    Ck := congrFun (congrFun (congrArg Prod.snd hl) 0) 0
  have t7 : W_Stage_online_1.generatedRoot.w_91_7 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_7
    Ck := congrFun (congrFun (congrArg Prod.snd hl) 0) 1
  have t8 : W_Stage_online_1.generatedRoot.w_91_8 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_8
    Ck := congrFun (congrFun (congrArg Prod.snd hl) 1) 0
  have t9 : W_Stage_online_1.generatedRoot.w_91_9 Con = W_Stage_keys_online_1_1.generatedRoot.w_87_9
    Ck := congrFun (congrFun (congrArg Prod.snd hl) 1) 1
  exact ⟨t2, t3, t4, t5, t6, t7, t8, t9⟩

set_option maxHeartbeats 400000000 in
/-- The last smudge PRF has identical key rows in both stages, independently of rows and payloads.
  -/
theorem smudge_total_agree1 {backend : BackendContext} {hashModel : HashModel} {params :
  W_Stage_online_1.Params}
    {paramsK : W_Stage_keys_online_1_1.Params}
    {Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params} {oon : _}
    (Fon : W_Stage_online_1.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_1_1.generatedRoot.Facts Ck ok) (L : Link1 Con Ck)
    (hq1 : 1 < W_Stage_online_1.modulus_0) (layout : RegularLayout W_Stage_online_1.modulus_0)
    (hlayout : backend.regularLayout W_Stage_online_1.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    W_Stage_online_1.generatedRoot.w_174_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_168_0 Ck ∧
      W_Stage_online_1.generatedRoot.w_174_1 Con = W_Stage_keys_online_1_1.generatedRoot.w_168_1 Ck
      ∧ W_Stage_online_1.generatedRoot.w_174_2 Con = W_Stage_keys_online_1_1.generatedRoot.w_168_2
      Ck ∧ W_Stage_online_1.generatedRoot.w_174_3 Con =
      W_Stage_keys_online_1_1.generatedRoot.w_168_3 Ck := by
  have ha0 := Fon.c6.h140_0
  rw [L.e14] at ha0
  have a0 : (W_Stage_online_1.generatedRoot.w_140_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_134_0 Ck) := (MxxExponentLut.hashSample_eq ha0).trans
    (MxxExponentLut.hashSample_eq Fk.c6.h134_0).symm
  have ha1 := Fon.c6.h141_0
  rw [L.e14] at ha1
  have a1 : (W_Stage_online_1.generatedRoot.w_141_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_135_0 Ck) := (MxxExponentLut.hashSample_eq ha1).trans
    (MxxExponentLut.hashSample_eq Fk.c6.h135_0).symm
  have ha2 := Fon.c6.h143_0
  rw [L.e14] at ha2
  have a2 : (W_Stage_online_1.generatedRoot.w_143_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_137_0 Ck) := (MxxExponentLut.hashSample_eq ha2).trans
    (MxxExponentLut.hashSample_eq Fk.c6.h137_0).symm
  have ha3 := Fon.c6.h144_0
  rw [L.e14] at ha3
  have a3 : (W_Stage_online_1.generatedRoot.w_144_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_138_0 Ck) := (MxxExponentLut.hashSample_eq ha3).trans
    (MxxExponentLut.hashSample_eq Fk.c6.h138_0).symm
  have k148 : ∀ j : Fin 16, Enc.A (Tup.ev (k := 6) (W_Stage_online_1.generatedRoot.w_148_0 Con,
    (W_Stage_online_1.generatedRoot.w_148_1 Con, (W_Stage_online_1.generatedRoot.w_148_2 Con,
    (W_Stage_online_1.generatedRoot.w_148_3 Con, (W_Stage_online_1.generatedRoot.w_148_4 Con,
    (W_Stage_online_1.generatedRoot.w_148_5 Con, ())))))) j) =
      Enc.A (Tup.ev (k := 6) (W_Stage_keys_online_1_1.generatedRoot.w_142_0 Ck,
        (W_Stage_keys_online_1_1.generatedRoot.w_142_1 Ck,
        (W_Stage_keys_online_1_1.generatedRoot.w_142_2 Ck,
        (W_Stage_keys_online_1_1.generatedRoot.w_142_3 Ck,
        (W_Stage_keys_online_1_1.generatedRoot.w_142_4 Ck,
        (W_Stage_keys_online_1_1.generatedRoot.w_142_5 Ck, ())))))) j) := by
    intro j
    obtain ⟨_, _, _, -, -, -, -, -, -, -, -, -, hc⟩ := Fon.c6.h148_5 j
    obtain ⟨_, _, _, -, -, -, -, -, -, -, -, -, hc'⟩ := Fk.c6.h142_5 j
    refine (PrfCellFlat.keys (by norm_num) hq1 layout hlayout hexact
      (W_Stage_online_1.parallel_generatedRoot_148.sem hc)).trans
      (Eq.trans ?_ (PrfCellFlat.keys (by norm_num) hq1 layout hlayout hexact
        (W_Stage_keys_online_1_1.parallel_generatedRoot_142.sem hc')).symm)
    rw [L.e139, L.e14, a0, a1, a2, a3, L.e70, L.e146, L.e147]
  have k149 : ∀ o : Fin 8, Enc.A (W_Stage_online_1.generatedRoot.w_149_0 Con o,
    (W_Stage_online_1.generatedRoot.w_149_1 Con o, (W_Stage_online_1.generatedRoot.w_149_2 Con o,
    (W_Stage_online_1.generatedRoot.w_149_3 Con o, (W_Stage_online_1.generatedRoot.w_149_4 Con o,
    (W_Stage_online_1.generatedRoot.w_149_5 Con o, ())))))) = Enc.A
    (W_Stage_keys_online_1_1.generatedRoot.w_143_0 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_143_1 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_143_2 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_143_3 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_143_4 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_143_5 Ck o, ())))))) := by
    intro o
    rw [rowsum_keys (W_Stage_online_1.parallel_generatedRoot_149.sem (Fon.c6.h149_5 o)),
      rowsum_keys (W_Stage_keys_online_1_1.parallel_generatedRoot_143.sem (Fk.c6.h143_5 o))]
    exact Finset.sum_congr rfl fun j _ ↦ k148 _
  have f149_0 : W_Stage_online_1.generatedRoot.w_149_0 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_143_0 Ck := funext fun o ↦ congrFun (congrFun (k149 o)
    0) 0
  have f149_1 : W_Stage_online_1.generatedRoot.w_149_1 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_143_1 Ck := funext fun o ↦ congrFun (congrFun (k149 o)
    0) 1
  have f149_2 : W_Stage_online_1.generatedRoot.w_149_2 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_143_2 Ck := funext fun o ↦ congrFun (congrFun (k149 o)
    1) 0
  have f149_3 : W_Stage_online_1.generatedRoot.w_149_3 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_143_3 Ck := funext fun o ↦ congrFun (congrFun (k149 o)
    1) 1
  have k152 : ∀ j : Fin 16, Enc.A (Tup.ev (k := 6) (W_Stage_online_1.generatedRoot.w_152_0 Con,
    (W_Stage_online_1.generatedRoot.w_152_1 Con, (W_Stage_online_1.generatedRoot.w_152_2 Con,
    (W_Stage_online_1.generatedRoot.w_152_3 Con, (W_Stage_online_1.generatedRoot.w_152_4 Con,
    (W_Stage_online_1.generatedRoot.w_152_5 Con, ())))))) j) =
      Enc.A (Tup.ev (k := 6) (W_Stage_keys_online_1_1.generatedRoot.w_146_0 Ck,
        (W_Stage_keys_online_1_1.generatedRoot.w_146_1 Ck,
        (W_Stage_keys_online_1_1.generatedRoot.w_146_2 Ck,
        (W_Stage_keys_online_1_1.generatedRoot.w_146_3 Ck,
        (W_Stage_keys_online_1_1.generatedRoot.w_146_4 Ck,
        (W_Stage_keys_online_1_1.generatedRoot.w_146_5 Ck, ())))))) j) := by
    intro j
    obtain ⟨_, _, _, -, -, -, -, -, -, -, -, -, hc⟩ := Fon.c6.h152_5 j
    obtain ⟨_, _, _, -, -, -, -, -, -, -, -, -, hc'⟩ := Fk.c6.h146_5 j
    refine (PrfCellFlat.keys_1 (by norm_num) hq1 layout hlayout hexact
      (W_Stage_online_1.parallel_generatedRoot_152.sem hc) (by omega)).trans
      (Eq.trans ?_ (PrfCellFlat.keys_1 (by norm_num) hq1 layout hlayout hexact
        (W_Stage_keys_online_1_1.parallel_generatedRoot_146.sem hc') (by omega)).symm)
    rw [L.e138, L.e14, f149_0, f149_2, f149_1, f149_3, L.e70, L.e150, L.e151]
  have k153 : ∀ o : Fin 8, Enc.A (W_Stage_online_1.generatedRoot.w_153_0 Con o,
    (W_Stage_online_1.generatedRoot.w_153_1 Con o, (W_Stage_online_1.generatedRoot.w_153_2 Con o,
    (W_Stage_online_1.generatedRoot.w_153_3 Con o, (W_Stage_online_1.generatedRoot.w_153_4 Con o,
    (W_Stage_online_1.generatedRoot.w_153_5 Con o, ())))))) = Enc.A
    (W_Stage_keys_online_1_1.generatedRoot.w_147_0 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_147_1 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_147_2 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_147_3 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_147_4 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_147_5 Ck o, ())))))) := by
    intro o
    rw [rowsum_keys (W_Stage_online_1.parallel_generatedRoot_153.sem (Fon.c6.h153_5 o)),
      rowsum_keys (W_Stage_keys_online_1_1.parallel_generatedRoot_147.sem (Fk.c6.h147_5 o))]
    exact Finset.sum_congr rfl fun j _ ↦ k152 _
  have f153_0 : W_Stage_online_1.generatedRoot.w_153_0 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_147_0 Ck := funext fun o ↦ congrFun (congrFun (k153 o)
    0) 0
  have f153_1 : W_Stage_online_1.generatedRoot.w_153_1 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_147_1 Ck := funext fun o ↦ congrFun (congrFun (k153 o)
    0) 1
  have f153_2 : W_Stage_online_1.generatedRoot.w_153_2 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_147_2 Ck := funext fun o ↦ congrFun (congrFun (k153 o)
    1) 0
  have f153_3 : W_Stage_online_1.generatedRoot.w_153_3 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_147_3 Ck := funext fun o ↦ congrFun (congrFun (k153 o)
    1) 1
  have k154 : ∀ o : Fin 8, Enc.A (W_Stage_online_1.generatedRoot.w_154_0 Con o,
    (W_Stage_online_1.generatedRoot.w_154_1 Con o, (W_Stage_online_1.generatedRoot.w_154_2 Con o,
    (W_Stage_online_1.generatedRoot.w_154_3 Con o, (W_Stage_online_1.generatedRoot.w_154_4 Con o,
    (W_Stage_online_1.generatedRoot.w_154_5 Con o, ())))))) = Enc.A
    (W_Stage_keys_online_1_1.generatedRoot.w_148_0 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_148_1 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_148_2 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_148_3 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_148_4 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_148_5 Ck o, ())))))) := by
    intro o
    refine (LutMemberFlat.keys (by norm_num) (L := 7) (by norm_num) hq1 layout hlayout hexact
        (fun i a b hb ↦ W_Stage_online_1.sequential_parallel_generatedRoot_154_18.sem hb)
          (W_Stage_online_1.parallel_generatedRoot_154.sem (Fon.c6.h154_5 o))).trans
      (Eq.trans ?_ (LutMemberFlat.keys (by norm_num) (L := 7) (by norm_num) hq1 layout hlayout
        hexact
        (fun i a b hb ↦ W_Stage_keys_online_1_1.sequential_parallel_generatedRoot_148_18.sem hb)
          (W_Stage_keys_online_1_1.parallel_generatedRoot_148.sem (Fk.c6.h148_5 o))).symm)
    rw [L.e132, L.e133, L.e134, L.e135, L.e14, f153_0, f153_2, f153_1, f153_3]
  have f154_0 : W_Stage_online_1.generatedRoot.w_154_0 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_148_0 Ck := funext fun o ↦ congrFun (congrFun (k154 o)
    0) 0
  have f154_1 : W_Stage_online_1.generatedRoot.w_154_1 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_148_1 Ck := funext fun o ↦ congrFun (congrFun (k154 o)
    0) 1
  have f154_2 : W_Stage_online_1.generatedRoot.w_154_2 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_148_2 Ck := funext fun o ↦ congrFun (congrFun (k154 o)
    1) 0
  have f154_3 : W_Stage_online_1.generatedRoot.w_154_3 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_148_3 Ck := funext fun o ↦ congrFun (congrFun (k154 o)
    1) 1
  have k155 : ∀ o : Fin 8, Enc.A (W_Stage_online_1.generatedRoot.w_155_0 Con o,
    (W_Stage_online_1.generatedRoot.w_155_1 Con o, (W_Stage_online_1.generatedRoot.w_155_2 Con o,
    (W_Stage_online_1.generatedRoot.w_155_3 Con o, (W_Stage_online_1.generatedRoot.w_155_4 Con o,
    (W_Stage_online_1.generatedRoot.w_155_5 Con o, ())))))) = Enc.A
    (W_Stage_keys_online_1_1.generatedRoot.w_149_0 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_149_1 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_149_2 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_149_3 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_149_4 Ck o,
    (W_Stage_keys_online_1_1.generatedRoot.w_149_5 Ck o, ())))))) := by
    intro o
    rw [placement_keys (W_Stage_online_1.parallel_generatedRoot_155.sem (Fon.c6.h155_5 o)),
      placement_keys (W_Stage_keys_online_1_1.parallel_generatedRoot_149.sem (Fk.c6.h149_5 o))]
    exact congrArg _ (k154 o)
  have toto := member_total_keys (M := 8) (by norm_num)
    (fams := (W_Stage_online_1.generatedRoot.w_155_0 Con, (W_Stage_online_1.generatedRoot.w_155_1
      Con, (W_Stage_online_1.generatedRoot.w_155_2 Con, (W_Stage_online_1.generatedRoot.w_155_3 Con,
      (W_Stage_online_1.generatedRoot.w_155_4 Con, (W_Stage_online_1.generatedRoot.w_155_5 Con,
      ())))))))
    (a := (W_Stage_online_1.generatedRoot.w_156_0 Con, (W_Stage_online_1.generatedRoot.w_159_0 Con,
      (W_Stage_online_1.generatedRoot.w_162_0 Con, (W_Stage_online_1.generatedRoot.w_165_0 Con,
      (W_Stage_online_1.generatedRoot.w_168_0 Con, (W_Stage_online_1.generatedRoot.w_171_0 Con,
      ())))))))
    (b := (W_Stage_online_1.generatedRoot.w_157_0 Con, (W_Stage_online_1.generatedRoot.w_160_0 Con,
      (W_Stage_online_1.generatedRoot.w_163_0 Con, (W_Stage_online_1.generatedRoot.w_166_0 Con,
      (W_Stage_online_1.generatedRoot.w_169_0 Con, (W_Stage_online_1.generatedRoot.w_172_0 Con,
      ())))))))
    (fun c ↦ by fin_cases c; exacts [Fon.c6.h156_0, Fon.c6.h159_0, Fon.c6.h162_0, Fon.c6.h165_0,
      Fon.c7.h168_0, Fon.c7.h171_0])
    (fun c ↦ by fin_cases c; exacts [Fon.c6.h157_0, Fon.c6.h160_0, Fon.c6.h163_0, Fon.c7.h166_0,
      Fon.c7.h169_0, Fon.c7.h172_0])
    (fun c ↦ by fin_cases c <;> rfl) Fon.c7.h174_5
    (fun i cur next hb ↦ W_Stage_online_1.sequential_generatedRoot_174.sem hb)
  have totk := member_total_keys (M := 8) (by norm_num)
    (fams := (W_Stage_keys_online_1_1.generatedRoot.w_149_0 Ck,
      (W_Stage_keys_online_1_1.generatedRoot.w_149_1 Ck,
      (W_Stage_keys_online_1_1.generatedRoot.w_149_2 Ck,
      (W_Stage_keys_online_1_1.generatedRoot.w_149_3 Ck,
      (W_Stage_keys_online_1_1.generatedRoot.w_149_4 Ck,
      (W_Stage_keys_online_1_1.generatedRoot.w_149_5 Ck, ())))))))
    (a := (W_Stage_keys_online_1_1.generatedRoot.w_150_0 Ck,
      (W_Stage_keys_online_1_1.generatedRoot.w_153_0 Ck,
      (W_Stage_keys_online_1_1.generatedRoot.w_156_0 Ck,
      (W_Stage_keys_online_1_1.generatedRoot.w_159_0 Ck,
      (W_Stage_keys_online_1_1.generatedRoot.w_162_0 Ck,
      (W_Stage_keys_online_1_1.generatedRoot.w_165_0 Ck, ())))))))
    (b := (W_Stage_keys_online_1_1.generatedRoot.w_151_0 Ck,
      (W_Stage_keys_online_1_1.generatedRoot.w_154_0 Ck,
      (W_Stage_keys_online_1_1.generatedRoot.w_157_0 Ck,
      (W_Stage_keys_online_1_1.generatedRoot.w_160_0 Ck,
      (W_Stage_keys_online_1_1.generatedRoot.w_163_0 Ck,
      (W_Stage_keys_online_1_1.generatedRoot.w_166_0 Ck, ())))))))
    (fun c ↦ by fin_cases c; exacts [Fk.c6.h150_0, Fk.c6.h153_0, Fk.c6.h156_0, Fk.c6.h159_0,
      Fk.c6.h162_0, Fk.c6.h165_0])
    (fun c ↦ by fin_cases c; exacts [Fk.c6.h151_0, Fk.c6.h154_0, Fk.c6.h157_0, Fk.c6.h160_0,
      Fk.c6.h163_0, Fk.c6.h166_0])
    (fun c ↦ by fin_cases c <;> rfl) Fk.c6.h168_5
    (fun i cur next hb ↦ W_Stage_keys_online_1_1.sequential_generatedRoot_168.sem hb)
  have ktot : Enc.A (W_Stage_online_1.generatedRoot.w_174_0 Con,
    (W_Stage_online_1.generatedRoot.w_174_1 Con, (W_Stage_online_1.generatedRoot.w_174_2 Con,
    (W_Stage_online_1.generatedRoot.w_174_3 Con, (W_Stage_online_1.generatedRoot.w_174_4 Con,
    (W_Stage_online_1.generatedRoot.w_174_5 Con, ())))))) =
      Enc.A (W_Stage_keys_online_1_1.generatedRoot.w_168_0 Ck,
        (W_Stage_keys_online_1_1.generatedRoot.w_168_1 Ck,
        (W_Stage_keys_online_1_1.generatedRoot.w_168_2 Ck,
        (W_Stage_keys_online_1_1.generatedRoot.w_168_3 Ck,
        (W_Stage_keys_online_1_1.generatedRoot.w_168_4 Ck,
        (W_Stage_keys_online_1_1.generatedRoot.w_168_5 Ck, ())))))) := by
    rw [toto, totk]
    exact Finset.sum_congr rfl fun j _ ↦ k155 _
  have t0 : W_Stage_online_1.generatedRoot.w_174_0 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_168_0 Ck := congrFun (congrFun ktot 0) 0
  have t1 : W_Stage_online_1.generatedRoot.w_174_1 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_168_1 Ck := congrFun (congrFun ktot 0) 1
  have t2 : W_Stage_online_1.generatedRoot.w_174_2 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_168_2 Ck := congrFun (congrFun ktot 1) 0
  have t3 : W_Stage_online_1.generatedRoot.w_174_3 Con =
    W_Stage_keys_online_1_1.generatedRoot.w_168_3 Ck := congrFun (congrFun ktot 1) 1
  exact ⟨t0, t1, t2, t3⟩

/-- The two state-output keys occurring literally in `online1_state_lut`. -/
noncomputable def stateKey1 {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_1.Params}
    (Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params) :
    Fin 2 → ExactMatrix W_Stage_online_1.modulus_0 8 1 25 :=
  ![W_Stage_online_1.generatedRoot.w_91_3 Con + rowOf
      (W_Stage_online_1.generatedRoot.w_101_0 Con * W_Stage_online_1.generatedRoot.w_119_0 Con) 1,
    W_Stage_online_1.generatedRoot.w_91_5 Con + rowOf
      (W_Stage_online_1.generatedRoot.w_101_0 Con * W_Stage_online_1.generatedRoot.w_119_0 Con) 2]

/-- The smudge PRF's component-1 keys after multiplication by the smudge digits. -/
noncomputable def smudgeKeyS1 {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_1.Params}
    (Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params) :
    Fin 2 → ExactMatrix W_Stage_online_1.modulus_0 8 1 1 :=
  fun k ↦ ![W_Stage_online_1.generatedRoot.w_174_1 Con,
    W_Stage_online_1.generatedRoot.w_174_3 Con] k * W_Stage_online_1.generatedRoot.w_182_0 Con

/-- A full-width slice of a single row is the original matrix. -/
theorem key1_slice_full {q columns : Nat} {input output : ExactMatrix q n 1 columns}
    (h : sliceMatrix input 0 1 0 columns output) : output = input := by
  rw [slice_rowOfG (by norm_num) h]
  funext row column
  have hrow : row = 0 := Subsingleton.elim _ _
  subst row
  rfl

set_option maxHeartbeats 400000000 in
/-- Every row of the keys-stage output is the online state-key and smudge-key combination. -/
theorem output_key_agree1 {backend : BackendContext} {hashModel : HashModel} {params :
  W_Stage_online_1.Params}
    {paramsK : W_Stage_keys_online_1_1.Params}
    {Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params} {oon : _}
    (Fon : W_Stage_online_1.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_1_1.generatedRoot.Facts Ck ok) (L : Link1 Con Ck)
    (hq1 : 1 < W_Stage_online_1.modulus_0) (layout : RegularLayout W_Stage_online_1.modulus_0)
    (hlayout : backend.regularLayout W_Stage_online_1.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    ∀ k : Fin 2, rowOf (W_Stage_keys_online_1_1.generatedRoot.w_185_0 Ck) k =
      stateKey1 Con k * W_Stage_online_1.generatedRoot.w_130_0 Con + smudgeKeyS1 Con k := by
  obtain ⟨loop2, loop3, loop4, loop5, loop6, loop7, loop8, loop9⟩ :=
    state_loop_agree1 Fon Fk L hq1 layout hlayout hexact
  obtain ⟨smudge0, smudge1, smudge2, smudge3⟩ :=
    smudge_total_agree1 Fon Fk L hq1 layout hlayout hexact
  have a4 : (W_Stage_online_1.generatedRoot.w_93_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_89_0 Ck) := rfl
  have ha3 := Fon.c4.h94_0
  rw [a4, L.e14] at ha3
  have a3 : (W_Stage_online_1.generatedRoot.w_94_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_90_0 Ck) := (MxxExponentLut.hashSample_eq ha3).trans
    (MxxExponentLut.hashSample_eq Fk.c4.h90_0).symm
  have ha5 := Fon.c4.h95_0
  rw [a4, L.e14] at ha5
  have a5 : (W_Stage_online_1.generatedRoot.w_95_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_91_0 Ck) := (MxxExponentLut.hashSample_eq ha5).trans
    (MxxExponentLut.hashSample_eq Fk.c4.h91_0).symm
  have ha6 := Fon.c4.h98_0
  rw [a4, L.e14] at ha6
  have a6 : (W_Stage_online_1.generatedRoot.w_98_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_94_0 Ck) := (MxxExponentLut.hashSample_eq ha6).trans
    (MxxExponentLut.hashSample_eq Fk.c4.h94_0).symm
  have ha7 := Fon.c4.h99_0
  rw [a4, L.e14] at ha7
  have a7 : (W_Stage_online_1.generatedRoot.w_99_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_95_0 Ck) := (MxxExponentLut.hashSample_eq ha7).trans
    (MxxExponentLut.hashSample_eq Fk.c4.h95_0).symm
  have ha16 := Fon.c0.h20_0
  rw [L.e14] at ha16
  have a16 : (W_Stage_online_1.generatedRoot.w_20_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_16_0 Ck) := (MxxExponentLut.hashSample_eq ha16).trans
    (MxxExponentLut.hashSample_eq Fk.c0.h16_0).symm
  have a17 : (W_Stage_online_1.generatedRoot.w_21_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_17_0 Ck) := rfl
  have a15 : (W_Stage_online_1.generatedRoot.w_22_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_18_0 Ck) := by
    unfold W_Stage_online_1.generatedRoot.w_22_0 W_Stage_keys_online_1_1.generatedRoot.w_18_0
    rw [a16, a17]
  have ha19 := Fon.c0.h23_0
  rw [L.e14] at ha19
  have a19 : (W_Stage_online_1.generatedRoot.w_23_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_19_0 Ck) := (MxxExponentLut.hashSample_eq ha19).trans
    (MxxExponentLut.hashSample_eq Fk.c0.h19_0).symm
  have a20 : (W_Stage_online_1.generatedRoot.w_24_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_20_0 Ck) := rfl
  have a18 : (W_Stage_online_1.generatedRoot.w_25_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_21_0 Ck) := by
    unfold W_Stage_online_1.generatedRoot.w_25_0 W_Stage_keys_online_1_1.generatedRoot.w_21_0
    rw [a19, a20]
  have ha22 := Fon.c0.h31_0
  rw [L.e14] at ha22
  have a22 : (W_Stage_online_1.generatedRoot.w_31_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_27_0 Ck) := (MxxExponentLut.hashSample_eq ha22).trans
    (MxxExponentLut.hashSample_eq Fk.c0.h27_0).symm
  have a23 : (W_Stage_online_1.generatedRoot.w_32_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_28_0 Ck) := rfl
  have a21 : (W_Stage_online_1.generatedRoot.w_33_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_29_0 Ck) := by
    unfold W_Stage_online_1.generatedRoot.w_33_0 W_Stage_keys_online_1_1.generatedRoot.w_29_0
    rw [a22, a23]
  have ha25 := Fon.c0.h34_0
  rw [L.e14] at ha25
  have a25 : (W_Stage_online_1.generatedRoot.w_34_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_30_0 Ck) := (MxxExponentLut.hashSample_eq ha25).trans
    (MxxExponentLut.hashSample_eq Fk.c0.h30_0).symm
  have a26 : (W_Stage_online_1.generatedRoot.w_35_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_31_0 Ck) := rfl
  have a24 : (W_Stage_online_1.generatedRoot.w_36_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_32_0 Ck) := by
    unfold W_Stage_online_1.generatedRoot.w_36_0 W_Stage_keys_online_1_1.generatedRoot.w_32_0
    rw [a25, a26]
  have ha30 := Fon.c0.h39_0
  rw [L.e14] at ha30
  have a30 : (W_Stage_online_1.generatedRoot.w_39_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_35_0 Ck) := (MxxExponentLut.hashSample_eq ha30).trans
    (MxxExponentLut.hashSample_eq Fk.c0.h35_0).symm
  have ha31 := Fon.c0.h40_0
  rw [L.e14] at ha31
  have a31 : (W_Stage_online_1.generatedRoot.w_40_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_36_0 Ck) := (MxxExponentLut.hashSample_eq ha31).trans
    (MxxExponentLut.hashSample_eq Fk.c0.h36_0).symm
  have ha29 := Fon.c0.h41_0
  rw [a30, a31] at ha29
  have a29 : (W_Stage_online_1.generatedRoot.w_41_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_37_0 Ck) := concatRows_unique ha29 Fk.c0.h37_0
  have ha28 := Fon.c0.h42_0
  rw [a29] at ha28
  have a28 : (W_Stage_online_1.generatedRoot.w_42_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_38_0 Ck) := gadgetDecomposeRuns_unique ha28 Fk.c0.h38_0
  have ha27 := Fon.c0.h43_0
  rw [a28] at ha27
  have a27 : (W_Stage_online_1.generatedRoot.w_43_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_39_0 Ck) := compactCenteredRebaseRuns_unique ha27
    Fk.c0.h39_0
  have a14 : (W_Stage_online_1.generatedRoot.w_53_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_49_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c1.h53_0, slice_rowOfG (by norm_num) Fk.c1.h49_0]
    exact (key1_stack_rows_agree Fon.c0.h22_0 Fon.c0.h26_0 Fon.c0.h33_0 Fon.c0.h37_0 Fon.c0.h38_0
      Fk.c0.h18_0 Fk.c0.h22_0 Fk.c0.h29_0 Fk.c0.h33_0 Fk.c0.h34_0 a15 a18 a21 a24 a27).1
  have a32 : (W_Stage_online_1.generatedRoot.w_54_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_50_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c1.h54_0, slice_rowOfG (by norm_num) Fk.c1.h50_0]
    exact (key1_stack_rows_agree Fon.c0.h22_0 Fon.c0.h26_0 Fon.c0.h33_0 Fon.c0.h37_0 Fon.c0.h38_0
      Fk.c0.h18_0 Fk.c0.h22_0 Fk.c0.h29_0 Fk.c0.h33_0 Fk.c0.h34_0 a15 a18 a21 a24 a27).2
  have ha37 := Fon.c1.h56_0
  rw [L.e14] at ha37
  have a37 : (W_Stage_online_1.generatedRoot.w_56_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_52_0 Ck) := (MxxExponentLut.hashSample_eq ha37).trans
    (MxxExponentLut.hashSample_eq Fk.c1.h52_0).symm
  have ha38 := Fon.c1.h57_0
  rw [L.e14] at ha38
  have a38 : (W_Stage_online_1.generatedRoot.w_57_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_53_0 Ck) := (MxxExponentLut.hashSample_eq ha38).trans
    (MxxExponentLut.hashSample_eq Fk.c1.h53_0).symm
  have ha36 := Fon.c1.h58_0
  rw [a37, a38] at ha36
  have a36 : (W_Stage_online_1.generatedRoot.w_58_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_54_0 Ck) := concatRows_unique ha36 Fk.c1.h54_0
  have ha35 := Fon.c1.h59_0
  rw [a36] at ha35
  have a35 : (W_Stage_online_1.generatedRoot.w_59_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_55_0 Ck) := gadgetDecomposeRuns_unique ha35 Fk.c1.h55_0
  have ha34 := Fon.c1.h60_0
  rw [a35] at ha34
  have a34 : (W_Stage_online_1.generatedRoot.w_60_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_56_0 Ck) := compactCenteredRebaseRuns_unique ha34
    Fk.c1.h56_0
  have a33 : (W_Stage_online_1.generatedRoot.w_66_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_62_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c1.h66_0, slice_rowOfG (by norm_num) Fk.c1.h62_0]
    exact (key1_stack_rows_agree Fon.c0.h22_0 Fon.c0.h26_0 Fon.c0.h33_0 Fon.c0.h37_0 Fon.c0.h38_0
      Fk.c0.h18_0 Fk.c0.h22_0 Fk.c0.h29_0 Fk.c0.h33_0 Fk.c0.h34_0 a15 a18 a21 a24 a34).1
  have a39 : (W_Stage_online_1.generatedRoot.w_67_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_63_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c2.h67_0, slice_rowOfG (by norm_num) Fk.c2.h63_0]
    exact (key1_stack_rows_agree Fon.c0.h22_0 Fon.c0.h26_0 Fon.c0.h33_0 Fon.c0.h37_0 Fon.c0.h38_0
      Fk.c0.h18_0 Fk.c0.h22_0 Fk.c0.h29_0 Fk.c0.h33_0 Fk.c0.h34_0 a15 a18 a21 a24 a34).2
  have ha43 := Fon.c2.h71_0
  rw [L.e70] at ha43
  have a43 : (W_Stage_online_1.generatedRoot.w_71_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_67_0 Ck) := (MxxExponentLut.hashSample_eq ha43).trans
    (MxxExponentLut.hashSample_eq Fk.c2.h67_0).symm
  have ha44 := Fon.c2.h73_0
  rw [L.e72] at ha44
  have a44 : (W_Stage_online_1.generatedRoot.w_73_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_69_0 Ck) := familyGetDynamic_unique ha44 Fk.c2.h69_0
  have ha42 := Fon.c2.h74_0
  rw [a43, a44] at ha42
  have a42 : (W_Stage_online_1.generatedRoot.w_74_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_70_0 Ck) := concatRows_unique ha42 Fk.c2.h70_0
  have ha41 := Fon.c2.h75_0
  rw [a42] at ha41
  have a41 : (W_Stage_online_1.generatedRoot.w_75_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_71_0 Ck) := gadgetDecomposeRuns_unique ha41 Fk.c2.h71_0
  have ha40 := Fon.c2.h76_0
  rw [a41] at ha40
  have a40 : (W_Stage_online_1.generatedRoot.w_76_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_72_0 Ck) := compactCenteredRebaseRuns_unique ha40
    Fk.c2.h72_0
  have a13 : (W_Stage_online_1.generatedRoot.w_79_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_75_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c2.h79_0, slice_rowOfG (by norm_num) Fk.c2.h75_0]
    exact (key1_stack_rows_agree Fon.c1.h53_0_1 Fon.c1.h55_0 Fon.c2.h66_0_1 Fon.c2.h68_0
      Fon.c2.h69_0 Fk.c1.h49_0_1 Fk.c1.h51_0 Fk.c2.h62_0_1 Fk.c2.h64_0 Fk.c2.h65_0 a14 a32 a33 a39
      a40).1
  have ha12 := Fon.c4.h103_0
  rw [a13] at ha12
  have a12 : (W_Stage_online_1.generatedRoot.w_103_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_99_0 Ck) := ringAutomorphismRuns_unique ha12
    Fk.c4.h99_0
  have a46 : (W_Stage_online_1.generatedRoot.w_80_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_76_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c3.h80_0, slice_rowOfG (by norm_num) Fk.c3.h76_0]
    exact (key1_stack_rows_agree Fon.c1.h53_0_1 Fon.c1.h55_0 Fon.c2.h66_0_1 Fon.c2.h68_0
      Fon.c2.h69_0 Fk.c1.h49_0_1 Fk.c1.h51_0 Fk.c2.h62_0_1 Fk.c2.h64_0 Fk.c2.h65_0 a14 a32 a33 a39
      a40).2
  have ha45 := Fon.c4.h104_0
  rw [a46] at ha45
  have a45 : (W_Stage_online_1.generatedRoot.w_104_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_100_0 Ck) := ringAutomorphismRuns_unique ha45
    Fk.c4.h100_0
  have ha52 := Fon.c3.h81_0
  rw [L.e70] at ha52
  have a52 : (W_Stage_online_1.generatedRoot.w_81_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_77_0 Ck) := (MxxExponentLut.hashSample_eq ha52).trans
    (MxxExponentLut.hashSample_eq Fk.c3.h77_0).symm
  have ha53 := Fon.c3.h83_0
  rw [L.e82] at ha53
  have a53 : (W_Stage_online_1.generatedRoot.w_83_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_79_0 Ck) := familyGetDynamic_unique ha53 Fk.c3.h79_0
  have ha51 := Fon.c3.h84_0
  rw [a52, a53] at ha51
  have a51 : (W_Stage_online_1.generatedRoot.w_84_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_80_0 Ck) := concatRows_unique ha51 Fk.c3.h80_0
  have ha50 := Fon.c3.h85_0
  rw [a51] at ha50
  have a50 : (W_Stage_online_1.generatedRoot.w_85_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_81_0 Ck) := gadgetDecomposeRuns_unique ha50 Fk.c3.h81_0
  have ha49 := Fon.c3.h86_0
  rw [a50] at ha49
  have a49 : (W_Stage_online_1.generatedRoot.w_86_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_82_0 Ck) := compactCenteredRebaseRuns_unique ha49
    Fk.c3.h82_0
  have a48 : (W_Stage_online_1.generatedRoot.w_89_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_85_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c3.h89_0, slice_rowOfG (by norm_num) Fk.c3.h85_0]
    exact (key1_stack_rows_agree Fon.c1.h53_0_1 Fon.c1.h55_0 Fon.c2.h66_0_1 Fon.c2.h68_0
      Fon.c2.h69_0 Fk.c1.h49_0_1 Fk.c1.h51_0 Fk.c2.h62_0_1 Fk.c2.h64_0 Fk.c2.h65_0 a14 a32 a33 a39
      a49).1
  have ha47 := Fon.c4.h107_0
  rw [a48] at ha47
  have a47 : (W_Stage_online_1.generatedRoot.w_107_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_103_0 Ck) := ringAutomorphismRuns_unique ha47
    Fk.c4.h103_0
  have a55 : (W_Stage_online_1.generatedRoot.w_90_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_86_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c3.h90_0, slice_rowOfG (by norm_num) Fk.c3.h86_0]
    exact (key1_stack_rows_agree Fon.c1.h53_0_1 Fon.c1.h55_0 Fon.c2.h66_0_1 Fon.c2.h68_0
      Fon.c2.h69_0 Fk.c1.h49_0_1 Fk.c1.h51_0 Fk.c2.h62_0_1 Fk.c2.h64_0 Fk.c2.h65_0 a14 a32 a33 a39
      a49).2
  have ha54 := Fon.c5.h108_0
  rw [a55] at ha54
  have a54 : (W_Stage_online_1.generatedRoot.w_108_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_104_0 Ck) := ringAutomorphismRuns_unique ha54
    Fk.c4.h104_0
  have ha58 := Fon.c5.h111_0
  rw [loop7, loop9] at ha58
  have a58 : (W_Stage_online_1.generatedRoot.w_111_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_107_0 Ck) := concatRows_unique ha58 Fk.c5.h107_0
  have ha57 := Fon.c5.h112_0
  rw [a58] at ha57
  have a57 : (W_Stage_online_1.generatedRoot.w_112_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_108_0 Ck) := gadgetDecomposeRuns_unique ha57
    Fk.c5.h108_0
  have ha56 := Fon.c5.h113_0
  rw [a57] at ha56
  have a56 : (W_Stage_online_1.generatedRoot.w_113_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_109_0 Ck) := compactCenteredRebaseRuns_unique ha56
    Fk.c5.h109_0
  have a11 : (W_Stage_online_1.generatedRoot.w_115_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_111_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c5.h115_0, slice_rowOfG (by norm_num) Fk.c5.h111_0]
    exact (key1_stack_rows_agree Fon.c4.h103_0_1 Fon.c4.h105_0 Fon.c5.h107_0_1 Fon.c5.h109_0
      Fon.c5.h110_0 Fk.c4.h99_0_1 Fk.c4.h101_0 Fk.c5.h103_0_1 Fk.c5.h105_0 Fk.c5.h106_0 a12 a45 a47
      a54 a56).1
  have a61 : (W_Stage_online_1.generatedRoot.w_116_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_112_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c5.h116_0, slice_rowOfG (by norm_num) Fk.c5.h112_0]
    exact (key1_stack_rows_agree Fon.c4.h103_0_1 Fon.c4.h105_0 Fon.c5.h107_0_1 Fon.c5.h109_0
      Fon.c5.h110_0 Fk.c4.h99_0_1 Fk.c4.h101_0 Fk.c5.h103_0_1 Fk.c5.h105_0 Fk.c5.h106_0 a12 a45 a47
      a54 a56).2
  have ha10 := Fon.c5.h117_0
  rw [a11, a61] at ha10
  have a10 : (W_Stage_online_1.generatedRoot.w_117_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_113_0 Ck) := concatRows_unique ha10 Fk.c5.h113_0
  have ha9 := Fon.c5.h118_0
  rw [a10] at ha9
  have a9 : (W_Stage_online_1.generatedRoot.w_118_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_114_0 Ck) := gadgetDecomposeRuns_unique ha9
    Fk.c5.h114_0
  have ha8 := Fon.c5.h119_0
  rw [a9] at ha8
  have a8 : (W_Stage_online_1.generatedRoot.w_119_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_115_0 Ck) := compactCenteredRebaseRuns_unique ha8
    Fk.c5.h115_0
  have a2 : (W_Stage_online_1.generatedRoot.w_201_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_117_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c8.h201_0, slice_rowOfG (by norm_num) Fk.c5.h117_0]
    exact (key1_stack_rows_agree Fon.c4.h94_0_1 Fon.c4.h96_0 Fon.c4.h98_0_1 Fon.c4.h100_0
      Fon.c4.h101_0 Fk.c4.h90_0_1 Fk.c4.h92_0 Fk.c4.h94_0_1 Fk.c4.h96_0 Fk.c4.h97_0 a3 a5 a6 a7
      a8).1
  have a0 : (W_Stage_online_1.generatedRoot.w_202_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_118_0 Ck) := by
    unfold W_Stage_online_1.generatedRoot.w_202_0 W_Stage_keys_online_1_1.generatedRoot.w_118_0
    rw [loop3, a2]
  have a64 : (W_Stage_online_1.generatedRoot.w_205_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_179_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c9.h205_0, slice_rowOfG (by norm_num) Fk.c7.h179_0]
    exact (key1_stack_rows_agree Fon.c4.h94_0_1 Fon.c4.h96_0 Fon.c4.h98_0_1 Fon.c4.h100_0
      Fon.c4.h101_0 Fk.c4.h90_0_1 Fk.c4.h92_0 Fk.c4.h94_0_1 Fk.c4.h96_0 Fk.c4.h97_0 a3 a5 a6 a7
      a8).2
  have a62 : (W_Stage_online_1.generatedRoot.w_206_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_180_0 Ck) := by
    unfold W_Stage_online_1.generatedRoot.w_206_0 W_Stage_keys_online_1_1.generatedRoot.w_180_0
    rw [loop5, a64]
  have a69 : (W_Stage_online_1.generatedRoot.w_125_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_119_0 Ck) := rfl
  have ha68 := Fon.c6.h126_0
  rw [a69] at ha68
  have a68 : (W_Stage_online_1.generatedRoot.w_126_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_120_0 Ck) := centeredRebaseRuns_unique ha68
    Fk.c5.h120_0
  have a70 : (W_Stage_online_1.generatedRoot.w_127_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_121_0 Ck) := rfl
  have a67 : (W_Stage_online_1.generatedRoot.w_128_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_122_0 Ck) := by
    unfold W_Stage_online_1.generatedRoot.w_128_0 W_Stage_keys_online_1_1.generatedRoot.w_122_0
    rw [a68, a70]
  have ha66 := Fon.c6.h129_0
  rw [a67] at ha66
  have a66 : (W_Stage_online_1.generatedRoot.w_129_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_123_0 Ck) := gadgetDecomposeRuns_unique ha66
    Fk.c5.h123_0
  have ha65 := Fon.c6.h130_0
  rw [a66] at ha65
  have a65 : (W_Stage_online_1.generatedRoot.w_130_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_124_0 Ck) := compactCenteredRebaseRuns_unique ha65
    Fk.c6.h124_0
  have a76 : (W_Stage_online_1.generatedRoot.w_176_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_170_0 Ck) := rfl
  have a75 : (W_Stage_online_1.generatedRoot.w_177_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_171_0 Ck) := by
    unfold W_Stage_online_1.generatedRoot.w_177_0 W_Stage_keys_online_1_1.generatedRoot.w_171_0
    rw [a76]
  have ha74 := Fon.c7.h178_0
  rw [a75] at ha74
  have a74 : (W_Stage_online_1.generatedRoot.w_178_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_172_0 Ck) := centeredRebaseRuns_unique ha74
    Fk.c7.h172_0
  have a77 : (W_Stage_online_1.generatedRoot.w_179_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_173_0 Ck) := rfl
  have a73 : (W_Stage_online_1.generatedRoot.w_180_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_174_0 Ck) := by
    unfold W_Stage_online_1.generatedRoot.w_180_0 W_Stage_keys_online_1_1.generatedRoot.w_174_0
    rw [a74, a77]
  have ha72 := Fon.c7.h181_0
  rw [a73] at ha72
  have a72 : (W_Stage_online_1.generatedRoot.w_181_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_175_0 Ck) := gadgetDecomposeRuns_unique ha72
    Fk.c7.h175_0
  have ha71 := Fon.c7.h182_0
  rw [a72] at ha71
  have a71 : (W_Stage_online_1.generatedRoot.w_182_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_176_0 Ck) := compactCenteredRebaseRuns_unique ha71
    Fk.c7.h176_0
  have state0 : stateKey1 Con 0 = W_Stage_keys_online_1_1.generatedRoot.w_118_0 Ck := by
    have hs := slice_rowOfG (by norm_num) Fon.c8.h201_0
    change W_Stage_online_1.generatedRoot.w_91_3 Con + rowOf
      (W_Stage_online_1.generatedRoot.w_120_0 Con) ⟨1, by decide⟩ = _
    rw [← hs]
    exact a0
  have state1 : stateKey1 Con 1 = W_Stage_keys_online_1_1.generatedRoot.w_180_0 Ck := by
    have hs := slice_rowOfG (by norm_num) Fon.c9.h205_0
    change W_Stage_online_1.generatedRoot.w_91_5 Con + rowOf
      (W_Stage_online_1.generatedRoot.w_120_0 Con) ⟨2, by decide⟩ = _
    rw [← hs]
    exact a62
  have smudgeRow0 := key1_slice_full Fk.c7.h169_0
  have smudgeRow1 := key1_slice_full Fk.c7.h182_0
  have rows := concatRows_two Fk.c7.h185_0
  intro k
  fin_cases k
  · change rowOf (W_Stage_keys_online_1_1.generatedRoot.w_185_0 Ck) 0 =
      stateKey1 Con 0 * W_Stage_online_1.generatedRoot.w_130_0 Con + smudgeKeyS1 Con 0
    rw [rows.1]
    change W_Stage_keys_online_1_1.generatedRoot.w_118_0 Ck *
      W_Stage_keys_online_1_1.generatedRoot.w_124_0 Ck +
      W_Stage_keys_online_1_1.generatedRoot.w_169_0 Ck *
        W_Stage_keys_online_1_1.generatedRoot.w_176_0 Ck = _
    rw [state0, a65]
    simp only [smudgeKeyS1, Fin.isValue, Matrix.cons_val_zero]
    rw [smudgeRow0, smudge1, a71]
  · change rowOf (W_Stage_keys_online_1_1.generatedRoot.w_185_0 Ck) 1 =
      stateKey1 Con 1 * W_Stage_online_1.generatedRoot.w_130_0 Con + smudgeKeyS1 Con 1
    rw [rows.2]
    change W_Stage_keys_online_1_1.generatedRoot.w_180_0 Ck *
      W_Stage_keys_online_1_1.generatedRoot.w_124_0 Ck +
      W_Stage_keys_online_1_1.generatedRoot.w_182_0 Ck *
        W_Stage_keys_online_1_1.generatedRoot.w_176_0 Ck = _
    rw [state1, a65]
    simp only [smudgeKeyS1, Fin.isValue, Matrix.cons_val_one, Matrix.cons_val_zero]
    rw [smudgeRow1, smudge3, a71]

set_option maxHeartbeats 400000000 in
/-- A solved last-step preimage supplies exactly `hpre` of `online1_output` for the anchor rows. -/
theorem online1_hpre {backend : BackendContext} {hashModel : HashModel} {params :
  W_Stage_online_1.Params}
    {paramsK : W_Stage_keys_online_1_1.Params}
    {Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params} {oon : _}
    (Fon : W_Stage_online_1.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_1_1.generatedRoot.Facts Ck ok) (L : Link1 Con Ck)
    (hq1 : 1 < W_Stage_online_1.modulus_0) (layout : RegularLayout W_Stage_online_1.modulus_0)
    (hlayout : backend.regularLayout W_Stage_online_1.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    ∀ {anchor : ExactMatrix W_Stage_online_1.modulus_0 8 2 54},
      anchor * W_Stage_online_1.generatedRoot.w_1_0 Con =
        W_Stage_keys_online_1_1.generatedRoot.w_185_0 Ck →
      ∀ k : Fin 2, rowOf anchor k * W_Stage_online_1.generatedRoot.w_1_0 Con =
        stateKey1 Con k * W_Stage_online_1.generatedRoot.w_130_0 Con + smudgeKeyS1 Con k := by
  intro anchor hsolved k
  rw [← rowOf_mul, hsolved]
  exact output_key_agree1 Fon Fk L hq1 layout hlayout hexact k

/-- The published last-step anchor is the public matrix of the preimage trapdoor. -/
theorem claim_anchor1 {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution) :
    execution.«stage_2».2.1 = execution.«stage_1».2.2.1 := by
  obtain ⟨-, -, h1, h2, -⟩ := h
  have hA := (anchors_trapdoor h1).2
  obtain ⟨_, hb⟩ := h2
  rw [show execution.«stage_2» = _ from hb]
  exact trapdoorSample_public hA

/-- The last online and keys contexts share precisely the `W_Claim` artifacts recorded in `Link1`;
the online preimage solves against the sampled last-step anchor and the keys-stage output. -/
theorem claim_link1 {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution) :
    ∃ (Con : W_Stage_online_1.generatedRoot.Ctx Backend.backend hashModel
        W_GeneratedClaim.stage_17_params)
      (Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx Backend.backend hashModel
        W_GeneratedClaim.stage_14_params),
      W_Stage_online_1.generatedRoot.Facts Con execution.«stage_17» ∧
      W_Stage_keys_online_1_1.generatedRoot.Facts Ck execution.«stage_14» ∧ Link1 Con Ck ∧
      execution.«stage_1».2.2.1 * W_Stage_online_1.generatedRoot.w_1_0 Con =
        W_Stage_keys_online_1_1.generatedRoot.w_185_0 Ck := by
  have hp := (claim_preimages h).2
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, -, h14, -, -, h17, -⟩ := h
  obtain ⟨wk, hk⟩ := h14
  obtain ⟨wo, ho⟩ := h17
  have Fk := W_Stage_keys_online_1_1.generatedRoot.facts ⟨_, wk⟩ hk
  have Fo := W_Stage_online_1.generatedRoot.facts ⟨_, wo⟩ ho
  refine ⟨⟨_, wo⟩, ⟨_, wk⟩, Fo, Fk, ⟨rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl,
    rfl, rfl, rfl, rfl, rfl, rfl⟩, ?_⟩
  have equation := hp.1
  rw [Fk.c9.hout] at equation
  exact equation

/-- Under the whole-claim wiring, the anchor rows and dispatched preimage discharge the last
online theorem's `hpre`, with its literal LUT keys and the smudge's component-1 keys. -/
theorem claim_hpreS1 {hashModel : HashModel} {external : GeneratedClaim.ExternalInputs}
    {execution : GeneratedClaim.Execution}
    (h : W_GeneratedClaim.Runs hashModel external execution)
    (hq1 : 1 < W_Stage_online_1.modulus_0) (layout : RegularLayout W_Stage_online_1.modulus_0)
    (hlayout : Backend.backend.regularLayout W_Stage_online_1.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    ∃ (Con : W_Stage_online_1.generatedRoot.Ctx Backend.backend hashModel
        W_GeneratedClaim.stage_17_params)
      (Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx Backend.backend hashModel
        W_GeneratedClaim.stage_14_params),
      W_Stage_online_1.generatedRoot.Facts Con execution.«stage_17» ∧
      W_Stage_keys_online_1_1.generatedRoot.Facts Ck execution.«stage_14» ∧ Link1 Con Ck ∧
      ∀ k : Fin 2, rowOf execution.«stage_1».2.2.1 k *
          W_Stage_online_1.generatedRoot.w_1_0 Con =
        stateKey1 Con k * W_Stage_online_1.generatedRoot.w_130_0 Con + smudgeKeyS1 Con k := by
  obtain ⟨Con, Ck, Fon, Fk, shared, solved⟩ := claim_link1 h
  exact ⟨Con, Ck, Fon, Fk, shared,
    online1_hpre Fon Fk shared hq1 layout hlayout hexact solved⟩


end Dio2Proof
