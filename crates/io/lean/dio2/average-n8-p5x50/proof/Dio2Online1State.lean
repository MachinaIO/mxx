import Dio2Online1Prf
import Dio2Lut
import Dio2LutBridge
import Dio2Final
import Dio2Part

/-!
# The state of step 1's online stage

The step Program reads the evaluator bit as `XB = X^2 · ONEPUB` (label member `1`, keys and rows
times `X^2`), multiplies it publicly into the state row and the product privately by the RHS of
`y` (`cell_flat`): the LUT input. The state LUT (`sequential_generatedRoot_91` and the last branch
at the root) is read at component `1` only (`lut_round_flat_last1`).
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen RefreshTopProof.GenIface
  RefreshTopProof.GenInst W_Stage_online_1 Mxx.ExponentLut.Refresh RefreshTopProof.GenFinal

/-- The tag prefix of the state root object (`rns-bgg/root-input/dio2/state`). -/
abbrev statePre : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 111, 111, 116, 45, 105, 110, 112, 117, 116, 47, 100,
    105, 111, 50, 47, 115, 116, 97, 116, 101]

/-- The tag prefix of the step-1 `y` RHS tops (`rns-bgg/rhs/RhsId(2)/0/1`). -/
abbrev rhsPreY : Blob :=
  [114, 110, 115, 45, 98, 103, 103, 47, 114, 104, 115, 47, 82, 104, 115, 73, 100, 40, 50, 41, 47,
    48, 47, 49]

set_option maxHeartbeats 32000000 in
/-- **The LUT input of step 1** (`(X² α₁) μ T` with the bound of `cell_flat`). -/
theorem online1_lut_input {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ α : ErrorPoly 8} (hα : polyNorm (Xe (n := 8) 2 * α) ≤ 1)
    (hαμ : polyNorm (Xe (n := 8) 2 * α * μ) ≤ 1)
    {G : ExactMatrix modulus_0 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G) {Ba : Nat}
    (hlab : Rel sm (labelKeysG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 labelPre 1)
      (fun j ↦ reducePoly modulus_0 8 (α * unitPair sm j) • G)
      ![generatedRoot.w_16_0 ctx 1, generatedRoot.w_27_0 ctx 1] Ba)
    (hpay : generatedRoot.w_47_0 ctx 1 = fun _ _ ↦ reducePoly modulus_0 8 α)
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm (keyPairG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 statePre [])
      (fun j ↦ reducePoly modulus_0 8 (μ * unitPair sp j) • g)
      ![generatedRoot.w_46_0 ctx, generatedRoot.w_63_0 ctx] Bs)
    {T : Fin 2 → ExactMatrix modulus_0 8 1 25} {eB : Fin 2 → ErrorMatrix 8 1 25} {BB : Nat}
    (hrhs : ∀ j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_70_0 ctx) modulus_0 25 rhsPreY (prfTopTags prfA1 prfB1 0) j +
        ![generatedRoot.w_72_0 ctx 0, generatedRoot.w_82_0 ctx 0] j = T j +
        reduceMatrix modulus_0 8 1 25 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![generatedRoot.w_79_0 ctx, generatedRoot.w_89_0 ctx],
        ![generatedRoot.w_80_0 ctx, generatedRoot.w_90_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (Xe (n := 8) 2 * α * μ) • T j)
      ![generatedRoot.w_78_0 ctx, generatedRoot.w_88_0 ctx]
      ((25 + 25) * 8 * ((25 + 25) * 8 * Ba * 2048 + 8 * 1 * Bs) * 2048 + 8 * 1 * BB) := by
  have hget : ∀ {f : Fin 2 → ExactMatrix modulus_0 8 1 25} {x : ExactMatrix modulus_0 8 1 25},
      familyGetStatic f 1 x → x = f 1 := by
    intro f x h
    obtain ⟨pos, hpos, rfl⟩ := h
    congr 1
    exact Fin.ext (by omega)
  have hgetp : ∀ {f : Fin 2 → ExactMatrix modulus_0 8 1 1} {x : ExactMatrix modulus_0 8 1 1},
      familyGetStatic f 1 x → x = f 1 := by
    intro f x h
    obtain ⟨pos, hpos, rfl⟩ := h
    congr 1
    exact Fin.ext (by omega)
  -- The label member's keys and rows.
  have k00 := keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le 1]) 0 0 (by norm_num) F.c0.h20_0
  have k10 := keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le 1]) 1 0 (by norm_num) F.c0.h23_0
  have k01 := keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le 1]) 0 1 (by norm_num) F.c0.h31_0
  have k11 := keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le 1]) 1 1 (by norm_num) F.c0.h34_0
  have r0 := hget F.c0.h17_0
  have r1 := hget F.c0.h28_0
  have rp := hgetp F.c0.h48_0
  -- `XB`: the label member times `X^2`.
  have hxb := rel_monomial_gen (by norm_num) hlab (k := 2) (by norm_num)
  have hxa : Rel sm ![![generatedRoot.w_22_0 ctx, generatedRoot.w_33_0 ctx],
      ![generatedRoot.w_25_0 ctx, generatedRoot.w_36_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (Xe (n := 8) 2 * α * unitPair sm j) • G)
      ![generatedRoot.w_19_0 ctx, generatedRoot.w_30_0 ctx] Ba := by
    have e : (fun a j ↦ multiplyMonomial
        (labelKeysG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 labelPre 1 a j) ((2 : Nat) : Int)) =
        ![![generatedRoot.w_22_0 ctx, generatedRoot.w_33_0 ctx],
          ![generatedRoot.w_25_0 ctx, generatedRoot.w_36_0 ctx]] := by
      funext a j
      fin_cases a <;> fin_cases j
      · rw [show generatedRoot.w_22_0 ctx = multiplyMonomial (generatedRoot.w_20_0 ctx) 2 from rfl, k00]
        rfl
      · rw [show generatedRoot.w_33_0 ctx = multiplyMonomial (generatedRoot.w_31_0 ctx) 2 from rfl, k01]
        rfl
      · rw [show generatedRoot.w_25_0 ctx = multiplyMonomial (generatedRoot.w_23_0 ctx) 2 from rfl, k10]
        rfl
      · rw [show generatedRoot.w_36_0 ctx = multiplyMonomial (generatedRoot.w_34_0 ctx) 2 from rfl, k11]
        rfl
    have er : (fun j ↦ multiplyMonomial (![generatedRoot.w_16_0 ctx 1, generatedRoot.w_27_0 ctx 1] j)
        ((2 : Nat) : Int)) = ![generatedRoot.w_19_0 ctx, generatedRoot.w_30_0 ctx] := by
      funext j
      fin_cases j
      · rw [show generatedRoot.w_19_0 ctx = multiplyMonomial (generatedRoot.w_17_0 ctx) 2 from rfl, r0]
        rfl
      · rw [show generatedRoot.w_30_0 ctx = multiplyMonomial (generatedRoot.w_28_0 ctx) 2 from rfl, r1]
        rfl
    have et : (fun j ↦ reducePoly modulus_0 8 (Xe (n := 8) 2) •
        (reducePoly modulus_0 8 (α * unitPair sm j) • G)) =
        fun j ↦ reducePoly modulus_0 8 (Xe (n := 8) 2 * α * unitPair sm j) • G := by
      funext j
      rw [smul_smul, ← map_mul, mul_assoc]
    rw [e, er, et] at hxb
    exact hxb
  -- The state's keys.
  have s00 := keyPairG_of_hash (pre := []) 0 0 (by norm_num) F.c0.h39_0
  have s10 := keyPairG_of_hash (pre := []) 1 0 (by norm_num) F.c0.h40_0
  have s01 := keyPairG_of_hash (pre := []) 0 1 (by norm_num) F.c1.h56_0
  have s11 := keyPairG_of_hash (pre := []) 1 1 (by norm_num) F.c1.h57_0
  have hks : keyPairG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 statePre [] =
      ![![generatedRoot.w_39_0 ctx, generatedRoot.w_56_0 ctx],
        ![generatedRoot.w_40_0 ctx, generatedRoot.w_57_0 ctx]] := by
    funext a j
    fin_cases a <;> fin_cases j
    exacts [s00.symm, s01.symm, s10.symm, s11.symm]
  rw [hks] at hs
  -- The payload `X² α`.
  have hpay' : generatedRoot.w_50_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 (Xe (n := 8) 2 * α) := by
    rw [show generatedRoot.w_50_0 ctx = multiplyMonomial (generatedRoot.w_48_0 ctx) 2 from rfl, rp, hpay,
      show (2 : Int) = ((2 : Nat) : Int) from rfl, multiplyMonomial_eq_gen _ (by norm_num)]
    funext r c
    simp [Matrix.smul_apply, map_mul]
  -- The RHS tops.
  have htop := prf_tops F.c2.h71_0 F.c3.h81_0
  have hb0 := familyGetDynamic_nat (i := 0) F.c2.h73_0 (by norm_num)
  have hb1 := familyGetDynamic_nat (i := 0) F.c3.h83_0 (by norm_num)
  have hrhs' : ∀ j, reducePoly modulus_0 8 sp • ![generatedRoot.w_71_0 ctx, generatedRoot.w_81_0 ctx] j +
      ![generatedRoot.w_73_0 ctx, generatedRoot.w_83_0 ctx] j = T j + reduceMatrix modulus_0 8 1 25 (eB j) := by
    rw [htop, hb0, hb1]; exact hrhs
  exact cell_flat (by norm_num) hq1 layout hlayout hexact hG hα hαμ hxa hpay'
    hg' hg hs F.c0.h22_0 F.c0.h26_0 F.c0.h33_0 F.c0.h37_0 F.c0.h38_0
    (Xs := ![generatedRoot.w_41_0 ctx, generatedRoot.w_58_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c0.h41_0, F.c1.h58_0])
    (D' := ![generatedRoot.w_42_0 ctx, generatedRoot.w_59_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c0.h42_0, F.c1.h59_0])
    (D := ![generatedRoot.w_43_0 ctx, generatedRoot.w_60_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c0.h43_0, F.c1.h60_0])
    (Y := ![generatedRoot.w_45_0 ctx, generatedRoot.w_62_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c0.h45_0, F.c1.h62_0])
    (A' := ![![generatedRoot.w_53_0 ctx, generatedRoot.w_66_0 ctx],
      ![generatedRoot.w_54_0 ctx, generatedRoot.w_67_0 ctx]])
    (fun j ↦ by fin_cases j; exacts [F.c1.h53_0, F.c1.h66_0])
    (fun j ↦ by fin_cases j; exacts [F.c1.h54_0, F.c2.h67_0])
    (c' := ![generatedRoot.w_52_0 ctx, generatedRoot.w_65_0 ctx]) (fun j ↦ by fin_cases j <;> rfl)
    F.c1.h53_0_1 F.c1.h55_0 F.c2.h66_0_1 F.c2.h68_0 F.c2.h69_0
    (Xb := ![generatedRoot.w_74_0 ctx, generatedRoot.w_84_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c2.h74_0, F.c3.h84_0])
    (E' := ![generatedRoot.w_75_0 ctx, generatedRoot.w_85_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c2.h75_0, F.c3.h85_0])
    (E := ![generatedRoot.w_76_0 ctx, generatedRoot.w_86_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c2.h76_0, F.c3.h86_0])
    hrhs' heB
    (X := ![generatedRoot.w_78_0 ctx, generatedRoot.w_88_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c2.h78_0, F.c3.h88_0])
    (Mk := ![![generatedRoot.w_79_0 ctx, generatedRoot.w_89_0 ctx],
      ![generatedRoot.w_80_0 ctx, generatedRoot.w_90_0 ctx]])
    (fun j ↦ by fin_cases j; exacts [F.c2.h79_0, F.c3.h89_0])
    (fun j ↦ by fin_cases j; exacts [F.c3.h80_0, F.c3.h90_0])

/-- The automorphisms of the step-1 state LUT (width four, stride two). -/
abbrev sigmas4 : List Int := [1, 3, 5, 15]

theorem sigmas4_odd : ∀ i : Fin 4, Odd (sigmaOf sigmas4 i).toNat := by
  decide

set_option maxHeartbeats 64000000 in
/-- **The step-1 state LUT, component 1.** The LUT input (`henc`, the cell of `online1_lut_input`
in LUT form) through the loop `sequential_generatedRoot_91` (three branches) and the last branch
at the root: `w_124 = s̄ K₀ + K₁ - Δ' ν g_q + e`. -/
theorem online1_state_lut {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} (hsm : polyNorm sm ≤ 1)
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {B : Nat}
    (henc : Rel sm ![![generatedRoot.w_79_0 ctx, generatedRoot.w_89_0 ctx],
        ![generatedRoot.w_80_0 ctx, generatedRoot.w_90_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ * unitPair sp j) • g)
      ![generatedRoot.w_78_0 ctx, generatedRoot.w_88_0 ctx] B)
    {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (keyPairG hashModel (generatedRoot.w_14_0 ctx) modulus_0 25 helperPre
        [.integer (sigmaOf sigmas4 i)])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution (sigmaOf sigmas4 i).toNat (sigmas4_odd i) sm) j) • g)
      ![generatedRoot.w_13_0 ctx i, generatedRoot.w_15_0 ctx i] Bh)
    {Δ' ν : ErrorPoly 8} {gq : ExactMatrix modulus_0 8 1 25} {C : Fin 4 → ExactPoly modulus_0 8}
    {BB : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : ErrorMatrix 8 1 25, CoeffBound e BB ∧
      reducePoly modulus_0 8 (substitution (sigmaOf sigmas4 i).toNat (sigmas4_odd i) sp) •
          generatedRoot.w_7_0 ctx i + generatedRoot.w_11_0 ctx i =
        (C i * reducePoly modulus_0 8 Δ') • gq + reduceMatrix modulus_0 8 1 25 e)
    (hμ : ∀ i, polyNorm (substitution (sigmaOf sigmas4 i).toNat (sigmas4_odd i) μ) ≤ 1)
    (hwalsh : ∑ i, reducePoly modulus_0 8 (substitution (sigmaOf sigmas4 i).toNat
      (sigmas4_odd i) μ) * C i = reducePoly modulus_0 8 ν) :
    ∃ e : ErrorMatrix 8 1 25, CoeffBound e
        (3 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
          ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 4 * (8 * 1 * BB)) ∧
      generatedRoot.w_124_0 ctx =
        reducePoly modulus_0 8 sm • (generatedRoot.w_91_3 ctx + rowOf (matrixMul
          (generatedRoot.w_101_0 ctx) (generatedRoot.w_119_0 ctx)) 1) +
          (generatedRoot.w_91_5 ctx + rowOf (matrixMul (generatedRoot.w_101_0 ctx)
            (generatedRoot.w_119_0 ctx)) 2) - reducePoly modulus_0 8 (Δ' * ν) • gq +
          reduceMatrix modulus_0 8 1 25 e := by
  let fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix modulus_0 8 1 25 :=
    ![![generatedRoot.w_5_0 ctx, generatedRoot.w_7_0 ctx],
      ![generatedRoot.w_9_0 ctx, generatedRoot.w_11_0 ctx]]
  have hloop := lut_loop_gen (n := 8) (S := ExactMatrix modulus_0 8 1 25 × ExactMatrix modulus_0 8 1 25 ×
      ExactMatrix modulus_0 8 1 25 × ExactMatrix modulus_0 8 1 25 × ExactMatrix modulus_0 8 1 25 ×
      ExactMatrix modulus_0 8 1 25 × ExactMatrix modulus_0 8 1 25 × ExactMatrix modulus_0 8 1 25 ×
      ExactMatrix modulus_0 8 1 25 × ExactMatrix modulus_0 8 1 25 × Unit) (W := 4) (sm := sm)
    (fun s ↦ ![![s.2.2.1, s.2.2.2.1], ![s.2.2.2.2.1, s.2.2.2.2.2.1]]) (fun s ↦ ![s.1, s.2.1])
    (fun s ↦ ![![s.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.1], ![s.2.2.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.2.2.1]])
    (lutBranchFlat (sigmaOf sigmas4) sigmas4_odd μ sp fam) fam
    ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) (by norm_num)
    (by
      have hk : (![![(0 : ExactMatrix modulus_0 8 1 25), 0], ![0, 0]] : Fin 2 → Fin 2 → _) = 0 := by
        funext k j; fin_cases k <;> fin_cases j <;> rfl
      have h0 : (![(0 : ExactMatrix modulus_0 8 1 25), 0] : Fin 2 → _) = 0 := by
        funext j; fin_cases j <;> rfl
      show Rel sm ![![(0 : ExactMatrix modulus_0 8 1 25), 0], ![0, 0]] 0 ![0, 0] 0
      rw [hk, h0]
      exact Rel.zero)
    (by
      funext k j
      fin_cases k <;> fin_cases j
      · exact familyGetDynamic_nat (i := 0) F.c0.h6_0 (by norm_num)
      · exact familyGetDynamic_nat (i := 0) F.c0.h8_0 (by norm_num)
      · exact familyGetDynamic_nat (i := 0) F.c0.h10_0 (by norm_num)
      · exact familyGetDynamic_nat (i := 0) F.c0.h12_0 (by norm_num))
    (fun i hi cur nxt Tc Bs hbody hsw hcur ↦ by
      have hstep := LutRoundStep.step (by norm_num) (log := 3) (by norm_num) hq1 layout hlayout hexact
        (sequential_generatedRoot_91.sem hbody) (by omega) (sigmas4_odd ⟨i, by omega⟩) hsm hg' hg henc
        (hhelp ⟨i, by omega⟩) hcur
      obtain ⟨hrel, hnext⟩ := hstep
      have e00 : cur.2.2.2.2.2.2.1 = fam 0 0 ⟨i, by omega⟩ := by
        simpa using congrFun (congrFun hsw 0) 0
      have e01 : cur.2.2.2.2.2.2.2.1 = fam 0 1 ⟨i, by omega⟩ := by
        simpa using congrFun (congrFun hsw 0) 1
      have e10 : cur.2.2.2.2.2.2.2.2.1 = fam 1 0 ⟨i, by omega⟩ := by
        simpa using congrFun (congrFun hsw 1) 0
      have e11 : cur.2.2.2.2.2.2.2.2.2.1 = fam 1 1 ⟨i, by omega⟩ := by
        simpa using congrFun (congrFun hsw 1) 1
      refine ⟨hrel.target_eq ?_, ?_⟩
      · funext j
        simp only [lutBranchFlat, dif_pos (show i < 4 by omega), branchFlat, Pi.add_apply]
        fin_cases j
        · simp only [Fin.zero_eta, Fin.isValue, Matrix.cons_val_zero, e00, e10]; rfl
        · simp only [Fin.mk_one, Fin.isValue, Matrix.cons_val_one, Matrix.cons_val_zero, e01,
            e11]; rfl
      · obtain ⟨n0, n1, n2, n3⟩ := hnext hi
        funext k j
        fin_cases k <;> fin_cases j
        · exact n0
        · exact n1
        · exact n2
        · exact n3)
    F.c3.h91_9
  obtain ⟨hrel, hsw⟩ := hloop
  simp only [show Int.toNat 3 = 3 from rfl] at hrel hsw
  have hs01 := congrFun (congrFun hsw 0) 1
  have hs11 := congrFun (congrFun hsw 1) 1
  simp only [Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one] at hs01 hs11
  have e92 := familyGetDynamic_nat (i := 3) F.c4.h92_0 (by norm_num)
  have e97 := familyGetDynamic_nat (i := 3) F.c4.h97_0 (by norm_num)
  have hkeys : ![![generatedRoot.w_94_0 ctx, generatedRoot.w_98_0 ctx],
      ![generatedRoot.w_95_0 ctx, generatedRoot.w_99_0 ctx]] = keyPairG hashModel
        (generatedRoot.w_14_0 ctx) modulus_0 25 helperPre [.integer (sigmaOf (W := 4) sigmas4 ⟨3, by omega⟩)] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact keyPairG_of_hash (pre := [.integer (sigmaOf (W := 4) sigmas4 ⟨3, by omega⟩)]) 0 0 (by norm_num) F.c4.h94_0
    · exact keyPairG_of_hash (pre := [.integer (sigmaOf (W := 4) sigmas4 ⟨3, by omega⟩)]) 0 1 (by norm_num) F.c4.h98_0
    · exact keyPairG_of_hash (pre := [.integer (sigmaOf (W := 4) sigmas4 ⟨3, by omega⟩)]) 1 0 (by norm_num) F.c4.h95_0
    · exact keyPairG_of_hash (pre := [.integer (sigmaOf (W := 4) sigmas4 ⟨3, by omega⟩)]) 1 1 (by norm_num) F.c4.h99_0
  have hhelpL := hhelp ⟨3, by omega⟩
  rw [← hkeys] at hhelpL
  have hround := lut_round_flat_last1 (by norm_num) (log := 3) (by norm_num) (L := 3) (W := 4) rfl hq1
    layout hlayout hexact (sigmaOf sigmas4) sigmas4_odd hsm hg' hg henc (fam := fam) hrel
    (h3 := ![generatedRoot.w_13_0 ctx ⟨3, by omega⟩, generatedRoot.w_15_0 ctx ⟨3, by omega⟩]) hhelpL
    (rc0 := generatedRoot.w_102_0 ctx) (rA00 := generatedRoot.w_103_0 ctx)
    (rA10 := generatedRoot.w_104_0 ctx) (rc1 := generatedRoot.w_106_0 ctx)
    (rA01 := generatedRoot.w_107_0 ctx) (rA11 := generatedRoot.w_108_0 ctx)
    F.c4.h102_0 F.c4.h103_0 F.c4.h104_0 F.c4.h106_0 F.c4.h107_0 F.c5.h108_0
    F.c4.h103_0_1 F.c4.h105_0 F.c5.h107_0_1 F.c5.h109_0 F.c5.h110_0
    (by show concatRows (fam 0 1 _) (fam 1 1 _) _; rw [← hs01, ← hs11]; exact F.c5.h111_0)
    F.c5.h112_0 F.c5.h113_0 F.c5.h115_0 F.c5.h116_0 F.c5.h117_0 F.c5.h118_0 F.c5.h119_0
    (by have h := F.c4.h94_0_1; rw [e92] at h; exact h) F.c4.h96_0
    (by have h := F.c4.h98_0_1; rw [e97] at h; exact h) F.c4.h100_0
    F.c4.h101_0 F.c5.h121_0 F.c6.h122_0 hswitch hμ hwalsh
  obtain ⟨e, he, heq⟩ := hround
  exact ⟨e, he, heq⟩

set_option maxHeartbeats 32000000 in
/-- **The output of step 1.** The state LUT's component `1` (`hlut`, from `online1_state_lut`)
re-targeted to the scalar form `Q_q` of the first prime (`w_131`), unmasked with the hint and the
output preimage, the smudge (`hs`, `hTH`) and the half `H`, and switched to the first prime: the
output holds the residues of `Δ' ν`. -/
theorem online1_output {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    (hRq : modulus_0 = 1606938044254428791916078459060479968194539524630727034176641 * modulus_1)
    {sm : ErrorPoly 8} {Δ' ν : ErrorPoly 8} {gq : ExactMatrix modulus_0 8 1 25}
    (hgq : gadgetMatrixRuns backend 4096 25 gq)
    {K0 K1 : ExactMatrix modulus_0 8 1 25} {el : ErrorMatrix 8 1 25} {Bl : Nat}
    (hel : CoeffBound el Bl)
    (hlut : generatedRoot.w_124_0 ctx = reducePoly modulus_0 8 sm • K0 + K1 -
      reducePoly modulus_0 8 (Δ' * ν) • gq + reduceMatrix modulus_0 8 1 25 el)
    {Bk : Fin 2 → ExactMatrix modulus_0 8 1 54} {eb : ErrorMatrix 8 1 54} {Beb : Nat}
    (hb : generatedRoot.w_0_0 ctx = maskPart (reducePoly modulus_0 8 sm) Bk +
      reduceMatrix modulus_0 8 1 54 eb) (heb : CoeffBound eb Beb)
    {Pd : ErrorMatrix 8 54 1} {BP : Nat}
    (hP : generatedRoot.w_1_0 ctx = reduceMatrix modulus_0 8 54 1 Pd) (hPd : CoeffBound Pd BP)
    {Ks : Fin 2 → ExactMatrix modulus_0 8 1 1}
    (hpre : ∀ k, Bk k * generatedRoot.w_1_0 ctx =
      ![K0, K1] k * generatedRoot.w_130_0 ctx + Ks k)
    {Tsm : ExactMatrix modulus_0 8 1 1} {es : ErrorMatrix 8 1 1} {Bs : Nat}
    (hs : generatedRoot.w_183_0 ctx = maskPart (reducePoly modulus_0 8 sm) Ks - Tsm +
      reduceMatrix modulus_0 8 1 1 es) (hes : CoeffBound es Bs)
    {M : ErrorMatrix 8 1 1} {BM : Nat}
    (hTH : Tsm - generatedRoot.w_186_0 ctx = reduceMatrix modulus_0 8 1 1 M) (hM : CoeffBound M BM)
    {DI : ErrorMatrix 8 25 1} {BD : Nat}
    (hDI : generatedRoot.w_130_0 ctx = reduceMatrix modulus_0 8 25 1 DI) (hDb : CoeffBound DI BD)
    (hnoise : 2 * (54 * 8 * Beb * BP + 25 * 8 * Bl * BD + Bs + BM) <
      1606938044254428791916078459060479968194539524630727034176641)
    (c : Fin 1) (i : Fin 8) :
    (generatedRoot.w_188_0 ctx 0 c).coeff i =
      (((reducePoly modulus_0 8 (Δ' * ν)).coeff i).val : ZMod modulus_1) := by
  have hn : (0 : Nat) < 8 := by norm_num
  have hq0 : 1 < modulus_0 := by rw [hRq]; norm_num [modulus_1]
  -- `gq` times the digits of `Q_q` is `Q_q`.
  have hD : generatedRoot.w_130_0 ctx = generatedRoot.w_129_0 ctx :=
    compactRebase_self_gen hn hq0 F.c6.h130_0
  have hGD := gadget_mul_decompose hq0 hn layout hlayout hexact hgq F.c6.h129_0
  have hQ : generatedRoot.w_128_0 ctx =
      fun _ _ ↦ ((1606938044254428791916078459060479968194539524630727034176641 : Int) :
        ExactPoly modulus_0 8) := by
    have e126 : generatedRoot.w_126_0 ctx = generatedRoot.w_125_0 ctx := by
      rw [centeredRebaseRuns_reduce F.c6.h126_0]
      exact reduceMatrix_centeredIntLift hq0 hn _
    show matrixMulScalarLeft (generatedRoot.w_126_0 ctx) (generatedRoot.w_127_0 ctx) = _
    rw [e126, matrixMulScalarLeft_eq]
    funext r c
    simp [generatedRoot.w_125_0, generatedRoot.w_127_0, matrixPolynomial_single]
  -- The re-targeted state.
  have hx : generatedRoot.w_131_0 ctx = maskPart (reducePoly modulus_0 8 sm)
      (fun k ↦ ![K0, K1] k * generatedRoot.w_130_0 ctx) -
      (((1606938044254428791916078459060479968194539524630727034176641 : Nat) : Int) :
        ExactPoly modulus_0 8) • reduceMatrix modulus_0 8 1 1 (fun _ _ ↦ Δ' * ν) +
      reduceMatrix modulus_0 8 1 1 (el * DI) := by
    show generatedRoot.w_124_0 ctx * generatedRoot.w_130_0 ctx = _
    have hgD : gq * generatedRoot.w_130_0 ctx = generatedRoot.w_128_0 ctx := by rw [hD, hGD]
    rw [hlut, Matrix.add_mul, Matrix.sub_mul, Matrix.add_mul, Matrix.smul_mul, Matrix.smul_mul,
      hgD, hQ, hDI, ← reduceMatrix_mul, ← hDI]
    simp only [maskPart, Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one]
    congr 2
    funext r c
    simp only [Matrix.smul_apply, smul_eq_mul, reduceMatrix, map_mul]
    push_cast
    ring
  have hζ : generatedRoot.w_187_0 ctx = matrixSub (matrixSub (matrixMul (generatedRoot.w_0_0 ctx)
      (generatedRoot.w_1_0 ctx)) (matrixAdd (generatedRoot.w_183_0 ctx) (generatedRoot.w_131_0 ctx)))
      (generatedRoot.w_186_0 ctx) := by
    show matrixSub (matrixSub (generatedRoot.w_3_0 ctx) (matrixAdd (generatedRoot.w_131_0 ctx)
      (generatedRoot.w_183_0 ctx))) (generatedRoot.w_186_0 ctx) = _
    rw [modulusReduce_same hn F.c0.h3_0]
    simp only [matrixAdd]
    rw [add_comm]
    rfl
  exact out_gen hn hRq (by norm_num) (by norm_num [modulus_1]) hb heb hP hPd
    (Kx := fun k ↦ ![K0, K1] k * generatedRoot.w_130_0 ctx) hpre hx (coeffBound_mul hn hel hDb)
    hs hes hTH hM hnoise hζ F.c7.h188_0 c i

end Dio2Proof
