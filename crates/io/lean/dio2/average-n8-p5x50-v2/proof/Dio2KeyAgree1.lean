import Dio2SmudgeHalf
import W_Stage_keys_online_1_1_Facts
import Gen_W_Stage_keys_online_1_1_Scopes
import Dio2Online1State
import Dio2KeysLoops

/-!
# Step-1 key agreement composition

The generated root proof follows every PRF bucket, LUT, corrected gather/placement,
packed column and indexed gadget digit in the online and zero/key runs.
`StateLink1` and `SmudgeLink1` contain shared imported-input equalities only;
no intermediate ciphertext or key equality is assumed by `output_key_agree1`.
`online1_root_hpre` transports the externally solved preimage equation to these keys.
-/
namespace Dio2Proof
open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen

alias online1_cell_keys := PrfCellFlat.keys_1W
alias online1_bucket_keys := bucket_keys
alias online1_lut_keys := LutMemberFlat.keys
alias online1_member_keys := member_total_keys
alias online1_column_keys := column_total_keys

/-- Congruence along any finite chain, with a shared key transformer at every step. -/
theorem key1_chain_agree {K : Type*} {count : Nat}
    (left right : Nat → K) (step : Nat → K → K)
    (hzero : left 0 = right 0)
    (hl : ∀ i, i < count → left (i + 1) = step i (left i))
    (hr : ∀ i, i < count → right (i + 1) = step i (right i)) :
    left count = right count := by
  have h : ∀ i, i ≤ count → left i = right i := by
    intro i
    induction i with
    | zero => intro _; exact hzero
    | succ i ih =>
      intro hi
      rw [hl i (by omega), hr i (by omega), ih (by omega)]
  exact h count le_rfl

/-- Preimage correctness transfers from the zero run once both key rows agree. -/
theorem online1_hpre {n q h m : Nat}
    {B : Fin 2 → ExactMatrix q n 1 h} {P : ExactMatrix q n h m}
    {zeroState zeroSmudge state smudge : Fin 2 → ExactMatrix q n 1 m}
    (hpre : ∀ k, B k * P = zeroState k + zeroSmudge k)
    (hstate : ∀ k, zeroState k = state k)
    (hsmudge : ∀ k, zeroSmudge k = smudge k) :
    ∀ k, B k * P = state k + smudge k := by
  intro k
  rw [hpre k, hstate k, hsmudge k]
/- Generated final-output adapter: gen_online1.py. -/
open Mxx.ExponentLut.Refresh RefreshTopProof
  RefreshTopProof.GenInst RefreshTopProof.GenFinal RefreshTopProof.GenIface
variable {n : Nat}

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

structure StateLink1 {backend : BackendContext} {hashModel : HashModel} {params :
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

set_option maxHeartbeats 400000000 in
theorem state_loop_agree1 {backend : BackendContext} {hashModel : HashModel} {params :
  W_Stage_online_1.Params}
    {paramsK : W_Stage_keys_online_1_1.Params}
    {Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params} {oon : _}
    (Fon : W_Stage_online_1.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_1_1.generatedRoot.Facts Ck ok) (L : StateLink1 Con Ck)
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
        online1HelperPre [.integer (MxxIR.selectInt (Int.ofNat i) [1, 3, 5, 15])]) x.2, famN (i + 1))
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

noncomputable def stateKey1 {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_1.Params}
    (Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params) :
    Fin 2 → ExactMatrix W_Stage_online_1.modulus_0 8 1 25 :=
  ![W_Stage_online_1.generatedRoot.w_91_3 Con + rowOf
      (W_Stage_online_1.generatedRoot.w_101_0 Con * W_Stage_online_1.generatedRoot.w_119_0 Con) 1,
    W_Stage_online_1.generatedRoot.w_91_5 Con + rowOf
      (W_Stage_online_1.generatedRoot.w_101_0 Con * W_Stage_online_1.generatedRoot.w_119_0 Con) 2]

/-- The actual digit-loop keys, before the final refresh combination. -/
noncomputable def smudgeKeyS1 {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_1.Params}
    (Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params) :
    Fin 2 → ExactMatrix W_Stage_online_1.modulus_0 8 1 1 :=
    ![W_Stage_online_1.generatedRoot.w_178_1 Con,
    W_Stage_online_1.generatedRoot.w_178_2 Con]

set_option maxHeartbeats 400000000 in
theorem output_key_agree1_of_smudge {backend : BackendContext} {hashModel : HashModel} {params :
  W_Stage_online_1.Params}
    {paramsK : W_Stage_keys_online_1_1.Params}
    {Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params} {oon : _}
    (Fon : W_Stage_online_1.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_1_1.generatedRoot.Facts Ck ok) (L : StateLink1 Con Ck)
    (hq1 : 1 < W_Stage_online_1.modulus_0) (layout : RegularLayout W_Stage_online_1.modulus_0)
    (hlayout : backend.regularLayout W_Stage_online_1.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    (hsmudge : ∀ port : Fin 2, smudgeKeyS1 Con port =
      ![W_Stage_keys_online_1_1.generatedRoot.w_172_1 Ck,
        W_Stage_keys_online_1_1.generatedRoot.w_172_2 Ck] port) :
    ∀ k : Fin 2, rowOf (W_Stage_keys_online_1_1.generatedRoot.w_178_0 Ck) k =
      stateKey1 Con k * W_Stage_online_1.generatedRoot.w_130_0 Con + smudgeKeyS1 Con k := by
  obtain ⟨loop2, loop3, loop4, loop5, loop6, loop7, loop8, loop9⟩ :=
    state_loop_agree1 Fon Fk L hq1 layout hlayout hexact
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
  have a2 : (W_Stage_online_1.generatedRoot.w_196_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_117_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c8.h196_0, slice_rowOfG (by norm_num) Fk.c5.h117_0]
    exact (key1_stack_rows_agree Fon.c4.h94_0_1 Fon.c4.h96_0 Fon.c4.h98_0_1 Fon.c4.h100_0
      Fon.c4.h101_0 Fk.c4.h90_0_1 Fk.c4.h92_0 Fk.c4.h94_0_1 Fk.c4.h96_0 Fk.c4.h97_0 a3 a5 a6 a7
      a8).1
  have a0 : (W_Stage_online_1.generatedRoot.w_197_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_118_0 Ck) := by
    unfold W_Stage_online_1.generatedRoot.w_197_0 W_Stage_keys_online_1_1.generatedRoot.w_118_0
    rw [loop3, a2]
  have a64 : (W_Stage_online_1.generatedRoot.w_200_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_174_0 Ck) := by
    rw [slice_rowOfG (by norm_num) Fon.c8.h200_0, slice_rowOfG (by norm_num) Fk.c7.h174_0]
    exact (key1_stack_rows_agree Fon.c4.h94_0_1 Fon.c4.h96_0 Fon.c4.h98_0_1 Fon.c4.h100_0
      Fon.c4.h101_0 Fk.c4.h90_0_1 Fk.c4.h92_0 Fk.c4.h94_0_1 Fk.c4.h96_0 Fk.c4.h97_0 a3 a5 a6 a7
      a8).2
  have a62 : (W_Stage_online_1.generatedRoot.w_201_0 Con) =
    (W_Stage_keys_online_1_1.generatedRoot.w_175_0 Ck) := by
    unfold W_Stage_online_1.generatedRoot.w_201_0 W_Stage_keys_online_1_1.generatedRoot.w_175_0
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
  have state0 : stateKey1 Con 0 = W_Stage_keys_online_1_1.generatedRoot.w_118_0 Ck := by
    have hs := slice_rowOfG (by norm_num) Fon.c8.h196_0
    change W_Stage_online_1.generatedRoot.w_91_3 Con + rowOf
      (W_Stage_online_1.generatedRoot.w_120_0 Con) ⟨1, by decide⟩ = _
    rw [← hs]
    exact a0
  have state1 : stateKey1 Con 1 = W_Stage_keys_online_1_1.generatedRoot.w_175_0 Ck := by
    have hs := slice_rowOfG (by norm_num) Fon.c8.h200_0
    change W_Stage_online_1.generatedRoot.w_91_5 Con + rowOf
      (W_Stage_online_1.generatedRoot.w_120_0 Con) ⟨2, by decide⟩ = _
    rw [← hs]
    exact a62
  have rows := concatRows_two Fk.c7.h178_0
  intro port
  fin_cases port
  · change rowOf (W_Stage_keys_online_1_1.generatedRoot.w_178_0 Ck) 0 = _
    rw [rows.1]
    change W_Stage_keys_online_1_1.generatedRoot.w_118_0 Ck *
      W_Stage_keys_online_1_1.generatedRoot.w_124_0 Ck +
      W_Stage_keys_online_1_1.generatedRoot.w_172_1 Ck = _
    rw [← state0, ← a65]
    exact congrArg (fun value ↦ stateKey1 Con 0 *
      W_Stage_online_1.generatedRoot.w_130_0 Con + value) (hsmudge 0).symm
  · change rowOf (W_Stage_keys_online_1_1.generatedRoot.w_178_0 Ck) 1 = _
    rw [rows.2]
    change W_Stage_keys_online_1_1.generatedRoot.w_175_0 Ck *
      W_Stage_keys_online_1_1.generatedRoot.w_124_0 Ck +
      W_Stage_keys_online_1_1.generatedRoot.w_172_2 Ck = _
    rw [← state1, ← a65]
    exact congrArg (fun value ↦ stateKey1 Con 1 *
      W_Stage_online_1.generatedRoot.w_130_0 Con + value) (hsmudge 1).symm
theorem online1_root_hpre_of_smudge {backend : BackendContext} {hashModel : HashModel} {params :
  W_Stage_online_1.Params}
    {paramsK : W_Stage_keys_online_1_1.Params}
    {Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params} {oon : _}
    (Fon : W_Stage_online_1.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_1_1.generatedRoot.Facts Ck ok) (L : StateLink1 Con Ck)
    (hq1 : 1 < W_Stage_online_1.modulus_0) (layout : RegularLayout W_Stage_online_1.modulus_0)
    (hlayout : backend.regularLayout W_Stage_online_1.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    (hsmudge : ∀ port : Fin 2, smudgeKeyS1 Con port =
      ![W_Stage_keys_online_1_1.generatedRoot.w_172_1 Ck,
        W_Stage_keys_online_1_1.generatedRoot.w_172_2 Ck] port) :
    ∀ {anchor : ExactMatrix W_Stage_online_1.modulus_0 8 2 54},
      anchor * W_Stage_online_1.generatedRoot.w_1_0 Con =
        W_Stage_keys_online_1_1.generatedRoot.w_178_0 Ck →
      ∀ k : Fin 2, rowOf anchor k * W_Stage_online_1.generatedRoot.w_1_0 Con =
        stateKey1 Con k * W_Stage_online_1.generatedRoot.w_130_0 Con + smudgeKeyS1 Con k := by
  intro anchor hsolved k
  rw [← rowOf_mul, hsolved]
  exact output_key_agree1_of_smudge Fon Fk L hq1 layout hlayout hexact hsmudge k

section

open W_Stage_keys_online_1_1 RefreshTopProof.GenInst RefreshTopProof.GenIface
/-- One exported digit step uses the digit-indexed weight and the matching packed blocks. -/
theorem key1_digit_step_graph {backend : MxxRuntime.BackendContext} {x_0 : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {inputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Fin 1 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 1)) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × Unit} {outputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 1) × Unit}
    (hq : 1 < modulus_0) (h : sequential_generatedRoot_172 backend x_0 params i_0 inputs outputs) (hr : i_0 < 1) :
    ∀ component : Fin 3, Tup.get (k := 3) outputs component = ![inputs.1, inputs.2.1, inputs.2.2.1] component +
      ![blk inputs.2.2.2.1 ⟨i_0, hr⟩, blk inputs.2.2.2.2.2.1 ⟨i_0, hr⟩,
        blk inputs.2.2.2.2.2.2.1 ⟨i_0, hr⟩] component *
        decG (inner := 25) backend 4096 25
          (inputs.2.2.2.2.1 ⟨i_0, hr⟩) := by
  obtain ⟨w_2_0, w_5_0, w_6_0, w_7_0, w_10_0, w_11_0, w_16_0, w_21_0, a0_0, a0_1, a0_2, a0_3, a0_4, a0_5, a0_6, a0_7, a0_8, a0_9, a0_10, a0_11, a0_12, a0_13, a0_14, a0_15, a0_16, a0_17, a0_18, a0_19, a0_20, a0_21, a0_22, a0_23, a0_24, a0_25, a0_26, a0_27, a0_28⟩ := h
  obtain ⟨_, hweight⟩ := familyGet_atG i_0 a0_9
  have hblock : w_6_0 = w_5_0 := by
    funext row column
    simpa only [Fin.isValue, Fin.val_zero, Nat.zero_mul, Nat.zero_add, ite_true]
      using a0_10.2 0 0 row column
  have hrebased : w_7_0 = w_5_0 :=
    (centeredRebaseRuns_unique a0_11 (centeredRebaseRuns_self hq (by norm_num) _)).trans hblock
  have hscalar : matrixMulScalarLeft w_7_0
      (matrixPolynomial [1]) = inputs.2.2.2.2.1 ⟨i_0, hr⟩ := by
    rw [hrebased, hweight]
    funext row column
    have hrow : row = 0 := Subsingleton.elim _ _
    have hcolumn : column = 0 := Subsingleton.elim _ _
    subst row column
    simp [matrixMulScalarLeft, matrixPolynomial_single]
  have hdigits : w_11_0 = decG (inner := 25) backend 4096 25
      (inputs.2.2.2.2.1 ⟨i_0, hr⟩) := by
    have hd := (RefreshTopProof.GenFinal.compactRebase_self_gen (by norm_num) hq a0_13).trans
      (decG_eq a0_12)
    simpa only [hscalar] using hd
  have hs0 : w_2_0 = blk inputs.2.2.2.1 ⟨i_0, hr⟩ := slice_blk ⟨i_0, hr⟩ a0_6 rfl
  have hs1 : w_16_0 = blk inputs.2.2.2.2.2.1 ⟨i_0, hr⟩ := slice_blk ⟨i_0, hr⟩ a0_20 rfl
  have hs2 : w_21_0 = blk inputs.2.2.2.2.2.2.1 ⟨i_0, hr⟩ := slice_blk ⟨i_0, hr⟩ a0_27 rfl
  rw [a0_28]
  intro component
  fin_cases component
  · change matrixAdd inputs.1 (matrixMul w_2_0 w_11_0) =
      inputs.1 + blk inputs.2.2.2.1 ⟨i_0, hr⟩ *
        decG (inner := 25) backend 4096 25
          (inputs.2.2.2.2.1 ⟨i_0, hr⟩)
    rw [hs0, hdigits]
    rfl
  · change matrixAdd inputs.2.1 (matrixMul w_16_0 w_11_0) =
      inputs.2.1 + blk inputs.2.2.2.2.2.1 ⟨i_0, hr⟩ *
        decG (inner := 25) backend 4096 25
          (inputs.2.2.2.2.1 ⟨i_0, hr⟩)
    rw [hs1, hdigits]
    rfl
  · change matrixAdd inputs.2.2.1 (matrixMul w_21_0 w_11_0) =
      inputs.2.2.1 + blk inputs.2.2.2.2.2.2.1 ⟨i_0, hr⟩ *
        decG (inner := 25) backend 4096 25
          (inputs.2.2.2.2.1 ⟨i_0, hr⟩)
    rw [hs2, hdigits]
    rfl

/-- The root smudge accumulator is the sum of all indexed digit actions. -/
theorem key1_digit_sums {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) (hq : 1 < modulus_0) :
    ∀ component : Fin 3, Tup.get (k := 3) ((generatedRoot.w_172_0 ctx, (generatedRoot.w_172_1 ctx, (generatedRoot.w_172_2 ctx, ())))) component =
      ∑ r : Fin 1, ![blk (generatedRoot.w_169_5 ctx) r, blk (generatedRoot.w_169_1 ctx) r,
        blk (generatedRoot.w_169_3 ctx) r] component *
      decG (inner := 25) backend 4096 25 (generatedRoot.w_171_0 ctx r) := by
  let term : Nat → Fin 3 → ExactMatrix modulus_0 8 1 1 := fun index component ↦
    if valid : index < 1 then
      let r : Fin 1 := ⟨index, valid⟩
      ![blk (generatedRoot.w_169_5 ctx) r, blk (generatedRoot.w_169_1 ctx) r,
        blk (generatedRoot.w_169_3 ctx) r] component *
      decG (inner := 25) backend 4096 25 (generatedRoot.w_171_0 ctx r)
    else 0
  have result := online1_digit_fold term (fun index current next hi hstep ↦ by
    change index < 1 at hi
    have hs := key1_digit_step_graph hq hstep hi
    intro component
    have hcurrent : Tup.get (k := 3) current component =
        ![current.1, current.2.1, current.2.2.1] component := by
      fin_cases component <;> rfl
    dsimp only [term]
    simp only [dif_pos hi, if_pos hi, hcurrent]
    exact hs component) (F.c7.h172_2)
  intro component
  simpa only [Finset.sum_range, term, dif_pos, Fin.is_lt] using result component

/-- Every exported digit uses its own radix power, not a repeated first-digit weight. -/
theorem key1_digit_weights {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) (r : Fin 1) :
    generatedRoot.w_171_0 ctx r = fun _ _ ↦ (904625697163682872125158399399175198226681342373069820760217620276403972497 : ExactPoly modulus_0 8) *
      (8 : ExactPoly modulus_0 8) ^ r.val := by
  have equation := (F.c7.h171_0 r).2.2
  change generatedRoot.w_171_0 ctx r = matrixMulScalarLeft
    (liftInteger ((8 : Int) ^ r.val)) (matrixPolynomial [(904625697163682872125158399399175198226681342373069820760217620276403972497)]) at equation
  rw [equation]
  funext row column
  simp only [matrixMulScalarLeft, liftInteger, matrixPolynomial_single, Int.cast_pow,
    Int.cast_natCast]
  ring

end

/-- Shared imported selectors, RHS rows and switching families used by the smudge key circuit.
No ciphertext rows or public-label payloads are equated. -/
structure SmudgeLink1 {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_1.Params} {paramsK : W_Stage_keys_online_1_1.Params}
    (Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params)
    (Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK) : Prop where
  e_w_133_0 : W_Stage_online_1.generatedRoot.w_133_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_127_0 Ck
  e_w_134_0 : W_Stage_online_1.generatedRoot.w_134_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_128_0 Ck
  e_w_135_0 : W_Stage_online_1.generatedRoot.w_135_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_129_0 Ck
  e_w_136_0 : W_Stage_online_1.generatedRoot.w_136_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_130_0 Ck
  e_w_139_0 : W_Stage_online_1.generatedRoot.w_139_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_133_0 Ck
  e_w_140_0 : W_Stage_online_1.generatedRoot.w_140_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_134_0 Ck
  e_w_147_0 : W_Stage_online_1.generatedRoot.w_147_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_141_0 Ck
  e_w_148_0 : W_Stage_online_1.generatedRoot.w_148_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_142_0 Ck
  e_w_151_0 : W_Stage_online_1.generatedRoot.w_151_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_145_0 Ck
  e_w_152_0 : W_Stage_online_1.generatedRoot.w_152_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_146_0 Ck

set_option maxHeartbeats 400000000 in
/-- Every exported smudge bucket and the readout compute the same keys in both runs. -/
theorem online1_readout_keys_agree {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_1.Params} {paramsK : W_Stage_keys_online_1_1.Params}
    {Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params} {oon : _}
    (Fon : W_Stage_online_1.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_1_1.generatedRoot.Facts Ck ok) (L : StateLink1 Con Ck)
    (S : SmudgeLink1 Con Ck) (hq : 1 < W_Stage_online_1.modulus_0)
    (layout : RegularLayout W_Stage_online_1.modulus_0)
    (hlayout : backend.regularLayout W_Stage_online_1.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    ∀ member : Fin 8, Enc.A ((W_Stage_online_1.generatedRoot.w_155_0 Con member, (W_Stage_online_1.generatedRoot.w_155_1 Con member, (W_Stage_online_1.generatedRoot.w_155_2 Con member, (W_Stage_online_1.generatedRoot.w_155_3 Con member, (W_Stage_online_1.generatedRoot.w_155_4 Con member, (W_Stage_online_1.generatedRoot.w_155_5 Con member, ()))))))) =
      Enc.A ((W_Stage_keys_online_1_1.generatedRoot.w_149_0 Ck member, (W_Stage_keys_online_1_1.generatedRoot.w_149_1 Ck member, (W_Stage_keys_online_1_1.generatedRoot.w_149_2 Ck member, (W_Stage_keys_online_1_1.generatedRoot.w_149_3 Ck member, (W_Stage_keys_online_1_1.generatedRoot.w_149_4 Ck member, (W_Stage_keys_online_1_1.generatedRoot.w_149_5 Ck member, ()))))))) := by
  have hash_input_0 := Fon.c6.h141_0
  rw [L.e14] at hash_input_0
  have key_input_0 : W_Stage_online_1.generatedRoot.w_141_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_135_0 Ck :=
    (hashSample_eq hash_input_0).trans (hashSample_eq Fk.c6.h135_0).symm
  have hash_input_1 := Fon.c6.h142_0
  rw [L.e14] at hash_input_1
  have key_input_1 : W_Stage_online_1.generatedRoot.w_142_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_136_0 Ck :=
    (hashSample_eq hash_input_1).trans (hashSample_eq Fk.c6.h136_0).symm
  have hash_input_2 := Fon.c6.h144_0
  rw [L.e14] at hash_input_2
  have key_input_2 : W_Stage_online_1.generatedRoot.w_144_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_138_0 Ck :=
    (hashSample_eq hash_input_2).trans (hashSample_eq Fk.c6.h138_0).symm
  have hash_input_3 := Fon.c6.h145_0
  rw [L.e14] at hash_input_3
  have key_input_3 : W_Stage_online_1.generatedRoot.w_145_0 Con = W_Stage_keys_online_1_1.generatedRoot.w_139_0 Ck :=
    (hashSample_eq hash_input_3).trans (hashSample_eq Fk.c6.h139_0).symm
  have cells_0 : ∀ member : Fin 16,
      Enc.A (Tup.ev (k := 6) ((W_Stage_online_1.generatedRoot.w_149_0 Con, (W_Stage_online_1.generatedRoot.w_149_1 Con, (W_Stage_online_1.generatedRoot.w_149_2 Con, (W_Stage_online_1.generatedRoot.w_149_3 Con, (W_Stage_online_1.generatedRoot.w_149_4 Con, (W_Stage_online_1.generatedRoot.w_149_5 Con, ()))))))) member) =
        Enc.A (Tup.ev (k := 6) ((W_Stage_keys_online_1_1.generatedRoot.w_143_0 Ck, (W_Stage_keys_online_1_1.generatedRoot.w_143_1 Ck, (W_Stage_keys_online_1_1.generatedRoot.w_143_2 Ck, (W_Stage_keys_online_1_1.generatedRoot.w_143_3 Ck, (W_Stage_keys_online_1_1.generatedRoot.w_143_4 Ck, (W_Stage_keys_online_1_1.generatedRoot.w_143_5 Ck, ()))))))) member) := by
    intro member
    obtain ⟨_, _, _, -, -, -, -, -, -, -, -, -, hc⟩ := Fon.c6.h149_5 member
    obtain ⟨_, _, _, -, -, -, -, -, -, -, -, -, hd⟩ := Fk.c6.h143_5 member
    refine (PrfCellFlat.keys (by norm_num) hq layout hlayout hexact
      (W_Stage_online_1.parallel_generatedRoot_149.sem hc)).trans
      (Eq.trans ?_ (PrfCellFlat.keys (by norm_num) hq layout hlayout hexact
        (W_Stage_keys_online_1_1.parallel_generatedRoot_143.sem hd)).symm)
    rw [S.e_w_140_0, L.e14, key_input_0, key_input_1, key_input_2, key_input_3, L.e70, S.e_w_147_0, S.e_w_148_0]
  have bucket_0 : ∀ member : Fin 8,
      Enc.A ((W_Stage_online_1.generatedRoot.w_150_0 Con member, (W_Stage_online_1.generatedRoot.w_150_1 Con member, (W_Stage_online_1.generatedRoot.w_150_2 Con member, (W_Stage_online_1.generatedRoot.w_150_3 Con member, (W_Stage_online_1.generatedRoot.w_150_4 Con member, (W_Stage_online_1.generatedRoot.w_150_5 Con member, ()))))))) =
        Enc.A ((W_Stage_keys_online_1_1.generatedRoot.w_144_0 Ck member, (W_Stage_keys_online_1_1.generatedRoot.w_144_1 Ck member, (W_Stage_keys_online_1_1.generatedRoot.w_144_2 Ck member, (W_Stage_keys_online_1_1.generatedRoot.w_144_3 Ck member, (W_Stage_keys_online_1_1.generatedRoot.w_144_4 Ck member, (W_Stage_keys_online_1_1.generatedRoot.w_144_5 Ck member, ()))))))) := by
    intro member
    rw [rowsum_keys (W_Stage_online_1.parallel_generatedRoot_150.sem
        (Fon.c6.h150_5 member)),
      rowsum_keys (W_Stage_keys_online_1_1.parallel_generatedRoot_144.sem
        (Fk.c6.h144_5 member))]
    exact Finset.sum_congr rfl fun cell _ ↦ cells_0 _
  have bucket_0_key_0 : W_Stage_online_1.generatedRoot.w_150_0 Con =
      W_Stage_keys_online_1_1.generatedRoot.w_144_0 Ck := funext fun member ↦
    congrFun (congrFun (bucket_0 member) 0) 0
  have bucket_0_key_1 : W_Stage_online_1.generatedRoot.w_150_1 Con =
      W_Stage_keys_online_1_1.generatedRoot.w_144_1 Ck := funext fun member ↦
    congrFun (congrFun (bucket_0 member) 0) 1
  have bucket_0_key_2 : W_Stage_online_1.generatedRoot.w_150_2 Con =
      W_Stage_keys_online_1_1.generatedRoot.w_144_2 Ck := funext fun member ↦
    congrFun (congrFun (bucket_0 member) 1) 0
  have bucket_0_key_3 : W_Stage_online_1.generatedRoot.w_150_3 Con =
      W_Stage_keys_online_1_1.generatedRoot.w_144_3 Ck := funext fun member ↦
    congrFun (congrFun (bucket_0 member) 1) 1
  have cells_1 : ∀ member : Fin 16,
      Enc.A (Tup.ev (k := 6) ((W_Stage_online_1.generatedRoot.w_153_0 Con, (W_Stage_online_1.generatedRoot.w_153_1 Con, (W_Stage_online_1.generatedRoot.w_153_2 Con, (W_Stage_online_1.generatedRoot.w_153_3 Con, (W_Stage_online_1.generatedRoot.w_153_4 Con, (W_Stage_online_1.generatedRoot.w_153_5 Con, ()))))))) member) =
        Enc.A (Tup.ev (k := 6) ((W_Stage_keys_online_1_1.generatedRoot.w_147_0 Ck, (W_Stage_keys_online_1_1.generatedRoot.w_147_1 Ck, (W_Stage_keys_online_1_1.generatedRoot.w_147_2 Ck, (W_Stage_keys_online_1_1.generatedRoot.w_147_3 Ck, (W_Stage_keys_online_1_1.generatedRoot.w_147_4 Ck, (W_Stage_keys_online_1_1.generatedRoot.w_147_5 Ck, ()))))))) member) := by
    intro member
    obtain ⟨_, _, _, -, -, -, -, -, -, -, -, -, hc⟩ := Fon.c6.h153_5 member
    obtain ⟨_, _, _, -, -, -, -, -, -, -, -, -, hd⟩ := Fk.c6.h147_5 member
    refine (PrfCellFlat.keys_1W (W := 2) (by norm_num) rfl hq layout hlayout hexact
      (W_Stage_online_1.parallel_generatedRoot_153.sem hc) (by have := member.isLt; omega)).trans
      (Eq.trans ?_ (PrfCellFlat.keys_1W (W := 2) (by norm_num) rfl hq layout hlayout hexact
        (W_Stage_keys_online_1_1.parallel_generatedRoot_147.sem hd) (by have := member.isLt; omega)).symm)
    rw [S.e_w_139_0, L.e14, bucket_0_key_0, bucket_0_key_1, bucket_0_key_2, bucket_0_key_3, L.e70, S.e_w_151_0, S.e_w_152_0]
  have bucket_1 : ∀ member : Fin 8,
      Enc.A ((W_Stage_online_1.generatedRoot.w_154_0 Con member, (W_Stage_online_1.generatedRoot.w_154_1 Con member, (W_Stage_online_1.generatedRoot.w_154_2 Con member, (W_Stage_online_1.generatedRoot.w_154_3 Con member, (W_Stage_online_1.generatedRoot.w_154_4 Con member, (W_Stage_online_1.generatedRoot.w_154_5 Con member, ()))))))) =
        Enc.A ((W_Stage_keys_online_1_1.generatedRoot.w_148_0 Ck member, (W_Stage_keys_online_1_1.generatedRoot.w_148_1 Ck member, (W_Stage_keys_online_1_1.generatedRoot.w_148_2 Ck member, (W_Stage_keys_online_1_1.generatedRoot.w_148_3 Ck member, (W_Stage_keys_online_1_1.generatedRoot.w_148_4 Ck member, (W_Stage_keys_online_1_1.generatedRoot.w_148_5 Ck member, ()))))))) := by
    intro member
    rw [rowsum_keys (W_Stage_online_1.parallel_generatedRoot_154.sem
        (Fon.c6.h154_5 member)),
      rowsum_keys (W_Stage_keys_online_1_1.parallel_generatedRoot_148.sem
        (Fk.c6.h148_5 member))]
    exact Finset.sum_congr rfl fun cell _ ↦ cells_1 _
  have bucket_1_key_0 : W_Stage_online_1.generatedRoot.w_154_0 Con =
      W_Stage_keys_online_1_1.generatedRoot.w_148_0 Ck := funext fun member ↦
    congrFun (congrFun (bucket_1 member) 0) 0
  have bucket_1_key_1 : W_Stage_online_1.generatedRoot.w_154_1 Con =
      W_Stage_keys_online_1_1.generatedRoot.w_148_1 Ck := funext fun member ↦
    congrFun (congrFun (bucket_1 member) 0) 1
  have bucket_1_key_2 : W_Stage_online_1.generatedRoot.w_154_2 Con =
      W_Stage_keys_online_1_1.generatedRoot.w_148_2 Ck := funext fun member ↦
    congrFun (congrFun (bucket_1 member) 1) 0
  have bucket_1_key_3 : W_Stage_online_1.generatedRoot.w_154_3 Con =
      W_Stage_keys_online_1_1.generatedRoot.w_148_3 Ck := funext fun member ↦
    congrFun (congrFun (bucket_1 member) 1) 1
  intro member
  refine (LutMemberFlat.keys (by norm_num) (L := 7) (by norm_num)
    hq layout hlayout hexact
    (fun index current next hbody ↦ W_Stage_online_1.sequential_parallel_generatedRoot_155_18.sem hbody)
    (W_Stage_online_1.parallel_generatedRoot_155.sem
      (Fon.c6.h155_5 member))).trans
    (Eq.trans ?_ (LutMemberFlat.keys (by norm_num) (L := 7) (by norm_num)
      hq layout hlayout hexact
      (fun index current next hbody ↦ W_Stage_keys_online_1_1.sequential_parallel_generatedRoot_149_18.sem hbody)
      (W_Stage_keys_online_1_1.parallel_generatedRoot_149.sem
        (Fk.c6.h149_5 member))).symm)
  rw [S.e_w_133_0, S.e_w_134_0, S.e_w_135_0, S.e_w_136_0, L.e14, bucket_1_key_0, bucket_1_key_1, bucket_1_key_2, bucket_1_key_3]

/-- Gathering is deterministic in its index; monomial placement acts on keys alone. -/
theorem online1_gather_monomial_keys {q n m members : Nat}
    {families : Tup (Fin members → ExactMatrix q n 1 m) 6}
    {gathered output : Tup (ExactMatrix q n 1 m) 6} {index : Int} {exponent : Nat}
    (hgather : GatherUnpack.Spec families index gathered)
    (hplace : ∀ component, output.get component =
      multiplyMonomial (gathered.get component) (exponent : Int)) :
    ∃ member : Fin members, member.val = index.toNat ∧
      Enc.A output = monoKeyF (exponent : Int) (Enc.A (Tup.ev families member)) := by
  obtain ⟨hbound, hnonneg, hgather⟩ := hgather
  let member : Fin members := ⟨index.toNat, hbound⟩
  have hsource : gathered = Tup.ev families member := by
    apply online1_tuple_ext
    intro component
    simpa only [Tup.get_ev] using hgather component
  have hout : output =
      (multiplyMonomial gathered.1 (exponent : Int),
      (multiplyMonomial gathered.2.1 (exponent : Int),
      (multiplyMonomial gathered.2.2.1 (exponent : Int),
      (multiplyMonomial gathered.2.2.2.1 (exponent : Int),
      (multiplyMonomial gathered.2.2.2.2.1 (exponent : Int),
      (multiplyMonomial gathered.2.2.2.2.2.1 (exponent : Int), ())))))) := by
    apply online1_tuple_ext
    intro component
    fin_cases component <;> exact hplace _
  refine ⟨member, rfl, ?_⟩
  simpa only [hsource] using monomial_keys hout

open W_Stage_online_1 in
/-- The actual fused placement exposes only its selected member's keys. -/
theorem online1_placement_keys {backend : MxxRuntime.BackendContext} {x_0 : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {inputs : (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × Unit} {outputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × Unit} (h : parallel_generatedRoot_156 backend x_0 params i_0 inputs outputs) :
    ∃ member : Fin 8, member.val = ((((Int.fdiv ((Int.ofNat i_0)) (8)) + (Int.fmod ((Int.ofNat i_0)) (8))))).toNat ∧
      Enc.A outputs = monoKeyF ((i_0 % 8 : Nat) : Int) (Enc.A (Tup.ev inputs member)) := by
  obtain ⟨w_2_0, w_6_0, w_10_0, w_14_0, w_18_0, w_22_0, a0_0, a0_1, a0_2, a0_3, a0_4, a0_5, a0_6, a0_7, a0_8, a0_9, a0_10, a0_11, a0_12, a0_13, a0_14, a0_15, a0_16, a0_17, a0_18, a0_19, a0_20, a0_21, a0_22, a0_23, a0_24, a0_25, a0_26⟩ := h
  apply online1_gather_monomial_keys
    (families := inputs) (gathered := (w_2_0, (w_6_0, (w_10_0, (w_14_0, (w_18_0, (w_22_0, ())))))))
    (index := (((Int.fdiv ((Int.ofNat i_0)) (8)) + (Int.fmod ((Int.ofNat i_0)) (8))))) (exponent := i_0 % 8)
  · apply GatherUnpack.sem (by omega) (by omega)
    intro component
    fin_cases component
    exacts [a0_4, a0_8, a0_12, a0_16, a0_20, a0_24]
  · intro component
    rw [a0_26]
    have hexponent : Int.fmod (Int.ofNat i_0) (8 : Int) = ((i_0 % 8 : Nat) : Int) :=
      fmod_ofNat i_0 8
    fin_cases component <;> rw [hexponent] <;> rfl

open W_Stage_online_1 in
/-- The actual exported member/column loop, conditional on its member relations. -/
theorem online1_packed_keys {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    :
    Enc.A ((generatedRoot.w_175_0 ctx), ((generatedRoot.w_175_1 ctx), ((generatedRoot.w_175_2 ctx), ((generatedRoot.w_175_3 ctx), ((generatedRoot.w_175_4 ctx), ((generatedRoot.w_175_5 ctx), ())))))) = memberKeyF fun member ↦ Enc.A (Tup.ev ((generatedRoot.w_156_0 ctx), ((generatedRoot.w_156_1 ctx), ((generatedRoot.w_156_2 ctx), ((generatedRoot.w_156_3 ctx), ((generatedRoot.w_156_4 ctx), ((generatedRoot.w_156_5 ctx), ())))))) member) := by
  exact member_total_keys (by norm_num : 2 ≤ 8)
    (fams := ((generatedRoot.w_156_0 ctx), ((generatedRoot.w_156_1 ctx), ((generatedRoot.w_156_2 ctx), ((generatedRoot.w_156_3 ctx), ((generatedRoot.w_156_4 ctx), ((generatedRoot.w_156_5 ctx), ())))))))
    (a := ((generatedRoot.w_157_0 ctx), ((generatedRoot.w_160_0 ctx), ((generatedRoot.w_163_0 ctx), ((generatedRoot.w_166_0 ctx), ((generatedRoot.w_169_0 ctx), ((generatedRoot.w_172_0 ctx), ()))))))) (b := ((generatedRoot.w_158_0 ctx), ((generatedRoot.w_161_0 ctx), ((generatedRoot.w_164_0 ctx), ((generatedRoot.w_167_0 ctx), ((generatedRoot.w_170_0 ctx), ((generatedRoot.w_173_0 ctx), ())))))))
    (fun component ↦ by fin_cases component; exacts [F.c6.h157_0, F.c6.h160_0, F.c6.h163_0, F.c6.h166_0, F.c7.h169_0, F.c7.h172_0])
    (fun component ↦ by fin_cases component; exacts [F.c6.h158_0, F.c6.h161_0, F.c6.h164_0, F.c7.h167_0, F.c7.h170_0, F.c7.h173_0])
    (fun component ↦ by fin_cases component <;> rfl)
    F.c7.h175_5
    (fun index current next hbody ↦ sequential_generatedRoot_175.sem hbody)

open W_Stage_keys_online_1_1 in
/-- The actual fused placement exposes only its selected member's keys. -/
theorem key1_placement_keys {backend : MxxRuntime.BackendContext} {x_0 : MxxRuntime.HashModel} {params : Params} {i_0 : Nat} {inputs : (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × (Fin 8 → (Mxx.Primitives.ExactMatrix modulus_0 8 1 25)) × Unit} {outputs : (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × (Mxx.Primitives.ExactMatrix modulus_0 8 1 25) × Unit} (h : parallel_generatedRoot_150 backend x_0 params i_0 inputs outputs) :
    ∃ member : Fin 8, member.val = ((((Int.fdiv ((Int.ofNat i_0)) (8)) + (Int.fmod ((Int.ofNat i_0)) (8))))).toNat ∧
      Enc.A outputs = monoKeyF ((i_0 % 8 : Nat) : Int) (Enc.A (Tup.ev inputs member)) := by
  obtain ⟨w_2_0, w_6_0, w_10_0, w_14_0, w_18_0, w_22_0, a0_0, a0_1, a0_2, a0_3, a0_4, a0_5, a0_6, a0_7, a0_8, a0_9, a0_10, a0_11, a0_12, a0_13, a0_14, a0_15, a0_16, a0_17, a0_18, a0_19, a0_20, a0_21, a0_22, a0_23, a0_24, a0_25, a0_26⟩ := h
  apply online1_gather_monomial_keys
    (families := inputs) (gathered := (w_2_0, (w_6_0, (w_10_0, (w_14_0, (w_18_0, (w_22_0, ())))))))
    (index := (((Int.fdiv ((Int.ofNat i_0)) (8)) + (Int.fmod ((Int.ofNat i_0)) (8))))) (exponent := i_0 % 8)
  · apply GatherUnpack.sem (by omega) (by omega)
    intro component
    fin_cases component
    exacts [a0_4, a0_8, a0_12, a0_16, a0_20, a0_24]
  · intro component
    rw [a0_26]
    have hexponent : Int.fmod (Int.ofNat i_0) (8 : Int) = ((i_0 % 8 : Nat) : Int) :=
      fmod_ofNat i_0 8
    fin_cases component <;> rw [hexponent] <;> rfl

open W_Stage_keys_online_1_1 in
/-- The actual exported member/column loop, conditional on its member relations. -/
theorem key1_packed_keys {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    :
    Enc.A ((generatedRoot.w_169_0 ctx), ((generatedRoot.w_169_1 ctx), ((generatedRoot.w_169_2 ctx), ((generatedRoot.w_169_3 ctx), ((generatedRoot.w_169_4 ctx), ((generatedRoot.w_169_5 ctx), ())))))) = memberKeyF fun member ↦ Enc.A (Tup.ev ((generatedRoot.w_150_0 ctx), ((generatedRoot.w_150_1 ctx), ((generatedRoot.w_150_2 ctx), ((generatedRoot.w_150_3 ctx), ((generatedRoot.w_150_4 ctx), ((generatedRoot.w_150_5 ctx), ())))))) member) := by
  exact member_total_keys (by norm_num : 2 ≤ 8)
    (fams := ((generatedRoot.w_150_0 ctx), ((generatedRoot.w_150_1 ctx), ((generatedRoot.w_150_2 ctx), ((generatedRoot.w_150_3 ctx), ((generatedRoot.w_150_4 ctx), ((generatedRoot.w_150_5 ctx), ())))))))
    (a := ((generatedRoot.w_151_0 ctx), ((generatedRoot.w_154_0 ctx), ((generatedRoot.w_157_0 ctx), ((generatedRoot.w_160_0 ctx), ((generatedRoot.w_163_0 ctx), ((generatedRoot.w_166_0 ctx), ()))))))) (b := ((generatedRoot.w_152_0 ctx), ((generatedRoot.w_155_0 ctx), ((generatedRoot.w_158_0 ctx), ((generatedRoot.w_161_0 ctx), ((generatedRoot.w_164_0 ctx), ((generatedRoot.w_167_0 ctx), ())))))))
    (fun component ↦ by fin_cases component; exacts [F.c6.h151_0, F.c6.h154_0, F.c6.h157_0, F.c6.h160_0, F.c6.h163_0, F.c6.h166_0])
    (fun component ↦ by fin_cases component; exacts [F.c6.h152_0, F.c6.h155_0, F.c6.h158_0, F.c6.h161_0, F.c6.h164_0, F.c6.h167_0])
    (fun component ↦ by fin_cases component <;> rfl)
    F.c6.h169_5
    (fun index current next hbody ↦ sequential_generatedRoot_169.sem hbody)

set_option maxHeartbeats 400000000 in
/-- Readout equality propagates through corrected gathering and packed column totals. -/
theorem online1_packed_keys_agree {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_1.Params} {paramsK : W_Stage_keys_online_1_1.Params}
    {Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params} {oon : _}
    (Fon : W_Stage_online_1.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_1_1.generatedRoot.Facts Ck ok) (L : StateLink1 Con Ck)
    (S : SmudgeLink1 Con Ck) (hq : 1 < W_Stage_online_1.modulus_0)
    (layout : RegularLayout W_Stage_online_1.modulus_0)
    (hlayout : backend.regularLayout W_Stage_online_1.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) (column : Fin 1) :
    Enc.A (blkT (C := 1) (m := 25) ((W_Stage_online_1.generatedRoot.w_175_0 Con), ((W_Stage_online_1.generatedRoot.w_175_1 Con), ((W_Stage_online_1.generatedRoot.w_175_2 Con), ((W_Stage_online_1.generatedRoot.w_175_3 Con), ((W_Stage_online_1.generatedRoot.w_175_4 Con), ((W_Stage_online_1.generatedRoot.w_175_5 Con), ())))))) column) =
      Enc.A (blkT (C := 1) (m := 25) ((W_Stage_keys_online_1_1.generatedRoot.w_169_0 Ck), ((W_Stage_keys_online_1_1.generatedRoot.w_169_1 Ck), ((W_Stage_keys_online_1_1.generatedRoot.w_169_2 Ck), ((W_Stage_keys_online_1_1.generatedRoot.w_169_3 Ck), ((W_Stage_keys_online_1_1.generatedRoot.w_169_4 Ck), ((W_Stage_keys_online_1_1.generatedRoot.w_169_5 Ck), ())))))) column) := by
  have hread := online1_readout_keys_agree Fon Fk L S hq layout hlayout hexact
  have hplace : ∀ member : Fin 8,
      Enc.A (((W_Stage_online_1.generatedRoot.w_156_0 Con member), ((W_Stage_online_1.generatedRoot.w_156_1 Con member), ((W_Stage_online_1.generatedRoot.w_156_2 Con member), ((W_Stage_online_1.generatedRoot.w_156_3 Con member), ((W_Stage_online_1.generatedRoot.w_156_4 Con member), ((W_Stage_online_1.generatedRoot.w_156_5 Con member), ()))))))) =
        Enc.A (((W_Stage_keys_online_1_1.generatedRoot.w_150_0 Ck member), ((W_Stage_keys_online_1_1.generatedRoot.w_150_1 Ck member), ((W_Stage_keys_online_1_1.generatedRoot.w_150_2 Ck member), ((W_Stage_keys_online_1_1.generatedRoot.w_150_3 Ck member), ((W_Stage_keys_online_1_1.generatedRoot.w_150_4 Ck member), ((W_Stage_keys_online_1_1.generatedRoot.w_150_5 Ck member), ()))))))) := by
    intro member
    obtain ⟨left, hleft, heqleft⟩ := online1_placement_keys
      (Fon.c6.h156_5 member)
    obtain ⟨right, hright, heqright⟩ := key1_placement_keys
      (Fk.c6.h150_5 member)
    have hsame : left = right := Fin.ext (hleft.trans hright.symm)
    rw [heqleft, heqright, hsame]
    exact congrArg (monoKeyF ((member.val % 8 : Nat) : Int)) (hread right)
  have hcolumn : column = 0 := Subsingleton.elim _ _
  subst column
  have htotal := (online1_packed_keys Con Fon).trans ((congrArg memberKeyF (funext hplace)).trans
      (key1_packed_keys Ck Fk).symm)
  have hblk : ∀ value : Tup (ExactMatrix W_Stage_online_1.modulus_0 8 1 25) 6,
      blkT (C := 1) (m := 25) value 0 = value := by
    intro value
    apply online1_tuple_ext
    intro component
    rw [blkT_get]
    funext row column
    simp [blk]
  rw [hblk, hblk]
  exact htotal
set_option maxHeartbeats 400000000 in
/-- The full exported smudge-key branch, including every indexed gadget digit. -/
theorem online1_smudge_keys_agree {backend : BackendContext} {hashModel : HashModel}
    {params : W_Stage_online_1.Params} {paramsK : W_Stage_keys_online_1_1.Params}
    {Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params} {oon : _}
    (Fon : W_Stage_online_1.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_1_1.generatedRoot.Facts Ck ok) (L : StateLink1 Con Ck)
    (S : SmudgeLink1 Con Ck) (hq : 1 < W_Stage_online_1.modulus_0)
    (layout : RegularLayout W_Stage_online_1.modulus_0)
    (hlayout : backend.regularLayout W_Stage_online_1.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0) :
    ∀ port : Fin 2, smudgeKeyS1 Con port =
      ![(W_Stage_keys_online_1_1.generatedRoot.w_172_1 Ck), (W_Stage_keys_online_1_1.generatedRoot.w_172_2 Ck)] port := by
  have hpacked := online1_packed_keys_agree Fon Fk L S hq layout hlayout hexact
  have hweight : ∀ r : Fin 1, (W_Stage_online_1.generatedRoot.w_177_0 Con) r =
      (W_Stage_keys_online_1_1.generatedRoot.w_171_0 Ck) r := by
    intro r
    rw [online1_digit_weights Con Fon r, key1_digit_weights Ck Fk r]
  intro port
  fin_cases port
  · change (W_Stage_online_1.generatedRoot.w_178_1 Con) = (W_Stage_keys_online_1_1.generatedRoot.w_172_1 Ck)
    have hleft := online1_digit_sums Con Fon hq 1
    have hright := key1_digit_sums Ck Fk hq 1
    change (W_Stage_online_1.generatedRoot.w_178_1 Con) = ∑ r : Fin 1, blk (C := 1) (m := 25) (W_Stage_online_1.generatedRoot.w_175_1 Con) r *
        decG (inner := 25) backend 4096 25
          ((W_Stage_online_1.generatedRoot.w_177_0 Con) r) at hleft
    change (W_Stage_keys_online_1_1.generatedRoot.w_172_1 Ck) = ∑ r : Fin 1, blk (C := 1) (m := 25) (W_Stage_keys_online_1_1.generatedRoot.w_169_1 Ck) r *
        decG (inner := 25) backend 4096 25
          ((W_Stage_keys_online_1_1.generatedRoot.w_171_0 Ck) r) at hright
    rw [hleft, hright]
    apply Finset.sum_congr rfl
    intro r _
    have hkey := congrFun (congrFun (hpacked r) 0) 1
    change blk (C := 1) (m := 25) (W_Stage_online_1.generatedRoot.w_175_1 Con) r = blk (C := 1) (m := 25) (W_Stage_keys_online_1_1.generatedRoot.w_169_1 Ck) r at hkey
    rw [hkey, hweight r]
  · change (W_Stage_online_1.generatedRoot.w_178_2 Con) = (W_Stage_keys_online_1_1.generatedRoot.w_172_2 Ck)
    have hleft := online1_digit_sums Con Fon hq 2
    have hright := key1_digit_sums Ck Fk hq 2
    change (W_Stage_online_1.generatedRoot.w_178_2 Con) = ∑ r : Fin 1, blk (C := 1) (m := 25) (W_Stage_online_1.generatedRoot.w_175_3 Con) r *
        decG (inner := 25) backend 4096 25
          ((W_Stage_online_1.generatedRoot.w_177_0 Con) r) at hleft
    change (W_Stage_keys_online_1_1.generatedRoot.w_172_2 Ck) = ∑ r : Fin 1, blk (C := 1) (m := 25) (W_Stage_keys_online_1_1.generatedRoot.w_169_3 Ck) r *
        decG (inner := 25) backend 4096 25
          ((W_Stage_keys_online_1_1.generatedRoot.w_171_0 Ck) r) at hright
    rw [hleft, hright]
    apply Finset.sum_congr rfl
    intro r _
    have hkey := congrFun (congrFun (hpacked r) 1) 1
    change blk (C := 1) (m := 25) (W_Stage_online_1.generatedRoot.w_175_3 Con) r = blk (C := 1) (m := 25) (W_Stage_keys_online_1_1.generatedRoot.w_169_3 Ck) r at hkey
    rw [hkey, hweight r]

/-- Root key agreement needs only shared imported inputs and the solved preimage. -/
theorem output_key_agree1 {backend : BackendContext} {hashModel : HashModel} {params :
  W_Stage_online_1.Params}
    {paramsK : W_Stage_keys_online_1_1.Params}
    {Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params} {oon : _}
    (Fon : W_Stage_online_1.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_1_1.generatedRoot.Facts Ck ok) (L : StateLink1 Con Ck) (S : SmudgeLink1 Con Ck)
    (hq1 : 1 < W_Stage_online_1.modulus_0) (layout : RegularLayout W_Stage_online_1.modulus_0)
    (hlayout : backend.regularLayout W_Stage_online_1.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    :
    ∀ k : Fin 2, rowOf (W_Stage_keys_online_1_1.generatedRoot.w_178_0 Ck) k =
      stateKey1 Con k * W_Stage_online_1.generatedRoot.w_130_0 Con + smudgeKeyS1 Con k := by
  exact output_key_agree1_of_smudge Fon Fk L hq1 layout hlayout hexact
    (online1_smudge_keys_agree Fon Fk L S hq1 layout hlayout hexact)

/-- Root key agreement needs only shared imported inputs and the solved preimage. -/
theorem online1_root_hpre {backend : BackendContext} {hashModel : HashModel} {params :
  W_Stage_online_1.Params}
    {paramsK : W_Stage_keys_online_1_1.Params}
    {Con : W_Stage_online_1.generatedRoot.Ctx backend hashModel params} {oon : _}
    (Fon : W_Stage_online_1.generatedRoot.Facts Con oon)
    {Ck : W_Stage_keys_online_1_1.generatedRoot.Ctx backend hashModel paramsK} {ok : _}
    (Fk : W_Stage_keys_online_1_1.generatedRoot.Facts Ck ok) (L : StateLink1 Con Ck) (S : SmudgeLink1 Con Ck)
    (hq1 : 1 < W_Stage_online_1.modulus_0) (layout : RegularLayout W_Stage_online_1.modulus_0)
    (hlayout : backend.regularLayout W_Stage_online_1.modulus_0 8 = some layout)
    (hexact : layout.droppedModuli = 0)
    :
    ∀ {anchor : ExactMatrix W_Stage_online_1.modulus_0 8 2 54},
      anchor * W_Stage_online_1.generatedRoot.w_1_0 Con =
        W_Stage_keys_online_1_1.generatedRoot.w_178_0 Ck →
      ∀ k : Fin 2, rowOf anchor k * W_Stage_online_1.generatedRoot.w_1_0 Con =
        stateKey1 Con k * W_Stage_online_1.generatedRoot.w_130_0 Con + smudgeKeyS1 Con k := by
  exact online1_root_hpre_of_smudge Fon Fk L hq1 layout hlayout hexact
    (online1_smudge_keys_agree Fon Fk L S hq1 layout hlayout hexact)

end Dio2Proof
