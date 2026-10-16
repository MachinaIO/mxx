import Dio2Online1State
import Dio2Online0Sigma
import W_Stage_online_0_Facts

/-!
# The state of step 0's online stage

As for step 1 (`Dio2Online1State`): the evaluator bit (`x₀ = 0`, so `XB = ONEPUB`, label member
`1`) times the state row and the RHS of `y₀` (`cell_flat`), then the state LUT
(`sequential_generatedRoot_317` and the last branch at the root, both components read).
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen RefreshTopProof.GenIface
  RefreshTopProof.GenInst W_Stage_online_0

set_option maxHeartbeats 32000000 in
/-- **The LUT input of step 0** (`α μ T` with the bound of `cell_flat`). -/
theorem online0_lut_input {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ α : ErrorPoly 8} (hα : polyNorm α ≤ 1) (hαμ : polyNorm (α * μ) ≤ 1)
    {G : ExactMatrix modulus_0 8 1 25} (hG : gadgetMatrixRuns backend 4096 25 G) {Ba : Nat}
    (hlab : Rel sm (labelKeysG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 labelPre 1)
      (fun j ↦ reducePoly modulus_0 8 (α * unitPair sm j) • G)
      ![generatedRoot.w_257_0 ctx 1, generatedRoot.w_262_0 ctx 1] Ba)
    (hpay : generatedRoot.w_276_0 ctx 1 = fun _ _ ↦ reducePoly modulus_0 8 α)
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm (keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 statePre [])
      (fun j ↦ reducePoly modulus_0 8 (μ * unitPair sp j) • g)
      ![generatedRoot.w_275_0 ctx, generatedRoot.w_290_0 ctx] Bs)
    {T : Fin 2 → ExactMatrix modulus_0 8 1 25} {eB : Fin 2 → ErrorMatrix 8 1 25} {BB : Nat}
    (hrhs : ∀ j, reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_26_0 ctx) modulus_0 25 rhsPreY (prfTopTags prfA0 prfB0 0) j +
        ![generatedRoot.w_298_0 ctx 0, generatedRoot.w_308_0 ctx 0] j = T j +
        reduceMatrix modulus_0 8 1 25 (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![generatedRoot.w_305_0 ctx, generatedRoot.w_315_0 ctx],
        ![generatedRoot.w_306_0 ctx, generatedRoot.w_316_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (α * μ) • T j)
      ![generatedRoot.w_304_0 ctx, generatedRoot.w_314_0 ctx]
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
  have k00 := keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le 1]) 0 0 (by norm_num) F.c6.h259_0
  have k10 := keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le 1]) 1 0 (by norm_num) F.c6.h260_0
  have k01 := keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le 1]) 0 1 (by norm_num) F.c6.h264_0
  have k11 := keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le 1]) 1 1 (by norm_num) F.c6.h265_0
  have r0 := hget F.c6.h258_0
  have r1 := hget F.c6.h263_0
  have rp := hgetp F.c7.h277_0
  have hxa : Rel sm ![![generatedRoot.w_259_0 ctx, generatedRoot.w_264_0 ctx],
      ![generatedRoot.w_260_0 ctx, generatedRoot.w_265_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (α * unitPair sm j) • G)
      ![generatedRoot.w_258_0 ctx, generatedRoot.w_263_0 ctx] Ba := by
    have e : labelKeysG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 labelPre 1 =
        ![![generatedRoot.w_259_0 ctx, generatedRoot.w_264_0 ctx],
          ![generatedRoot.w_260_0 ctx, generatedRoot.w_265_0 ctx]] := by
      funext a j
      fin_cases a <;> fin_cases j
      exacts [k00.symm, k01.symm, k10.symm, k11.symm]
    have er : (![generatedRoot.w_257_0 ctx 1, generatedRoot.w_262_0 ctx 1] : Fin 2 → _) =
        ![generatedRoot.w_258_0 ctx, generatedRoot.w_263_0 ctx] := by
      rw [r0, r1]
    rw [e, er] at hlab
    exact hlab
  have s00 := keyPairG_of_hash (pre := []) 0 0 (by norm_num) F.c7.h268_0
  have s10 := keyPairG_of_hash (pre := []) 1 0 (by norm_num) F.c7.h269_0
  have s01 := keyPairG_of_hash (pre := []) 0 1 (by norm_num) F.c7.h283_0
  have s11 := keyPairG_of_hash (pre := []) 1 1 (by norm_num) F.c8.h284_0
  have hks : keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 statePre [] =
      ![![generatedRoot.w_268_0 ctx, generatedRoot.w_283_0 ctx],
        ![generatedRoot.w_269_0 ctx, generatedRoot.w_284_0 ctx]] := by
    funext a j
    fin_cases a <;> fin_cases j
    exacts [s00.symm, s01.symm, s10.symm, s11.symm]
  rw [hks] at hs
  have hpay' : generatedRoot.w_277_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 α := by
    rw [rp, hpay]
  have htop := prf_tops F.c8.h297_0 F.c9.h307_0
  have hb0 := familyGetDynamic_nat (i := 0) F.c8.h299_0 (by norm_num)
  have hb1 := familyGetDynamic_nat (i := 0) F.c9.h309_0 (by norm_num)
  have hrhs' : ∀ j, reducePoly modulus_0 8 sp • ![generatedRoot.w_297_0 ctx, generatedRoot.w_307_0 ctx] j +
      ![generatedRoot.w_299_0 ctx, generatedRoot.w_309_0 ctx] j = T j + reduceMatrix modulus_0 8 1 25 (eB j) := by
    rw [htop, hb0, hb1]; exact hrhs
  exact cell_flat (by norm_num) hq1 layout hlayout hexact hG hα hαμ hxa hpay'
    hg' hg hs F.c6.h259_0_1 F.c6.h261_0 F.c6.h264_0_1 F.c7.h266_0 F.c7.h267_0
    (Xs := ![generatedRoot.w_270_0 ctx, generatedRoot.w_285_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c7.h270_0, F.c8.h285_0])
    (D' := ![generatedRoot.w_271_0 ctx, generatedRoot.w_286_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c7.h271_0, F.c8.h286_0])
    (D := ![generatedRoot.w_272_0 ctx, generatedRoot.w_287_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c7.h272_0, F.c8.h287_0])
    (Y := ![generatedRoot.w_274_0 ctx, generatedRoot.w_289_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c7.h274_0, F.c8.h289_0])
    (A' := ![![generatedRoot.w_280_0 ctx, generatedRoot.w_293_0 ctx],
      ![generatedRoot.w_281_0 ctx, generatedRoot.w_294_0 ctx]])
    (fun j ↦ by fin_cases j; exacts [F.c7.h280_0, F.c8.h293_0])
    (fun j ↦ by fin_cases j; exacts [F.c7.h281_0, F.c8.h294_0])
    (c' := ![generatedRoot.w_279_0 ctx, generatedRoot.w_292_0 ctx]) (fun j ↦ by fin_cases j <;> rfl)
    F.c7.h280_0_1 F.c7.h282_0 F.c8.h293_0_1 F.c8.h295_0 F.c8.h296_0
    (Xb := ![generatedRoot.w_300_0 ctx, generatedRoot.w_310_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c8.h300_0, F.c9.h310_0])
    (E' := ![generatedRoot.w_301_0 ctx, generatedRoot.w_311_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c8.h301_0, F.c9.h311_0])
    (E := ![generatedRoot.w_302_0 ctx, generatedRoot.w_312_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c9.h302_0, F.c9.h312_0])
    hrhs' heB
    (X := ![generatedRoot.w_304_0 ctx, generatedRoot.w_314_0 ctx])
    (fun j ↦ by fin_cases j; exacts [F.c9.h304_0, F.c10.h314_0])
    (Mk := ![![generatedRoot.w_305_0 ctx, generatedRoot.w_315_0 ctx],
      ![generatedRoot.w_306_0 ctx, generatedRoot.w_316_0 ctx]])
    (fun j ↦ by fin_cases j; exacts [F.c9.h305_0, F.c10.h315_0])
    (fun j ↦ by fin_cases j; exacts [F.c9.h306_0, F.c10.h316_0])

set_option maxHeartbeats 64000000 in
/-- **The step-0 state LUT.** The LUT input (`henc`) through the loop `sequential_generatedRoot_317`
(three branches) and the last branch at the root, both components: `w_350` (component 0) and
`w_370` (component 1) are `s̄ K₀ⱼ + K₁ⱼ - Δ' ν (s_p, 1)ⱼ g_q + eⱼ`. -/
theorem online0_state_lut {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs)
    (hq1 : 1 < modulus_0) (layout : RegularLayout modulus_0)
    (hlayout : backend.regularLayout modulus_0 8 = some layout) (hexact : layout.droppedModuli = 0)
    {sm sp μ : ErrorPoly 8} (hsm : polyNorm sm ≤ 1)
    {g' g : ExactMatrix modulus_0 8 1 25} (hg' : gadgetMatrixRuns backend 4096 25 g')
    (hg : centeredRebaseRuns g' g) {B : Nat}
    (henc : Rel sm ![![generatedRoot.w_305_0 ctx, generatedRoot.w_315_0 ctx],
        ![generatedRoot.w_306_0 ctx, generatedRoot.w_316_0 ctx]]
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) * μ * unitPair sp j) • g)
      ![generatedRoot.w_304_0 ctx, generatedRoot.w_314_0 ctx] B)
    {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (keyPairG hashModel (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre
        [.integer (sigmaOf sigmas4 i)])
      (fun j ↦ reducePoly modulus_0 8 (((1 : Nat) : Int) *
        unitPair (substitution (sigmaOf sigmas4 i).toNat (sigmas4_odd i) sm) j) • g)
      ![generatedRoot.w_255_0 ctx i, generatedRoot.w_256_0 ctx i] Bh)
    {Δ' ν : ErrorPoly 8} {gq : ExactMatrix modulus_0 8 1 25} {C : Fin 4 → ExactPoly modulus_0 8}
    {BB : Nat}
    (hswitch : ∀ (jc : Fin 2) (i : Fin 4), ∃ e : ErrorMatrix 8 1 25, CoeffBound e BB ∧
      reducePoly modulus_0 8 (substitution (sigmaOf sigmas4 i).toNat (sigmas4_odd i) sp) •
          ![![generatedRoot.w_247_0 ctx, generatedRoot.w_249_0 ctx],
            ![generatedRoot.w_251_0 ctx, generatedRoot.w_253_0 ctx]] 0 jc i +
        ![![generatedRoot.w_247_0 ctx, generatedRoot.w_249_0 ctx],
          ![generatedRoot.w_251_0 ctx, generatedRoot.w_253_0 ctx]] 1 jc i =
        (C i * reducePoly modulus_0 8 (Δ' * unitPair sp jc)) • gq + reduceMatrix modulus_0 8 1 25 e)
    (hμ : ∀ i, polyNorm (substitution (sigmaOf sigmas4 i).toNat (sigmas4_odd i) μ) ≤ 1)
    (hwalsh : ∑ i, reducePoly modulus_0 8 (substitution (sigmaOf sigmas4 i).toNat
      (sigmas4_odd i) μ) * C i = reducePoly modulus_0 8 ν) :
    (∃ e : ErrorMatrix 8 1 25, CoeffBound e
        (3 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
          ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 4 * (8 * 1 * BB)) ∧
      generatedRoot.w_350_0 ctx =
        reducePoly modulus_0 8 sm • (generatedRoot.w_317_2 ctx + rowOf (matrixMul
          (generatedRoot.w_327_0 ctx) (generatedRoot.w_345_0 ctx)) 1) +
          (generatedRoot.w_317_4 ctx + rowOf (matrixMul (generatedRoot.w_327_0 ctx)
            (generatedRoot.w_345_0 ctx)) 2) - reducePoly modulus_0 8 (Δ' * ν * unitPair sp 0) • gq +
          reduceMatrix modulus_0 8 1 25 e) ∧
    (∃ e : ErrorMatrix 8 1 25, CoeffBound e
        (3 * ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) +
          ((25 + 25) * 8 * Bh * 2048 + (25 + 25) * 8 * B * 2048) + 4 * (8 * 1 * BB)) ∧
      generatedRoot.w_370_0 ctx =
        reducePoly modulus_0 8 sm • (generatedRoot.w_317_3 ctx + rowOf (matrixMul
          (generatedRoot.w_327_0 ctx) (generatedRoot.w_365_0 ctx)) 1) +
          (generatedRoot.w_317_5 ctx + rowOf (matrixMul (generatedRoot.w_327_0 ctx)
            (generatedRoot.w_365_0 ctx)) 2) - reducePoly modulus_0 8 (Δ' * ν * unitPair sp 1) • gq +
          reduceMatrix modulus_0 8 1 25 e) := by
  let fam : Fin 2 → Fin 2 → Fin 4 → ExactMatrix modulus_0 8 1 25 :=
    ![![generatedRoot.w_247_0 ctx, generatedRoot.w_249_0 ctx],
      ![generatedRoot.w_251_0 ctx, generatedRoot.w_253_0 ctx]]
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
      · exact familyGetDynamic_nat (i := 0) F.c6.h248_0 (by norm_num)
      · exact familyGetDynamic_nat (i := 0) F.c6.h250_0 (by norm_num)
      · exact familyGetDynamic_nat (i := 0) F.c6.h252_0 (by norm_num)
      · exact familyGetDynamic_nat (i := 0) F.c6.h254_0 (by norm_num))
    (fun i hi cur nxt Tc Bs hbody hsw hcur ↦ by
      have hstep := LutRoundStep.step (by norm_num) (log := 3) (by norm_num) hq1 layout hlayout hexact
        (sequential_generatedRoot_317.sem hbody) (by omega) (sigmas4_odd ⟨i, by omega⟩) hsm hg' hg henc
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
    F.c10.h317_9
  obtain ⟨hrel, hsw⟩ := hloop
  simp only [show Int.toNat 3 = 3 from rfl] at hrel hsw
  have hs00 := congrFun (congrFun hsw 0) 0
  have hs10 := congrFun (congrFun hsw 1) 0
  have hs01 := congrFun (congrFun hsw 0) 1
  have hs11 := congrFun (congrFun hsw 1) 1
  simp only [Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one] at hs00 hs10 hs01 hs11
  have e318 := familyGetDynamic_nat (i := 3) F.c10.h318_0 (by norm_num)
  have e323 := familyGetDynamic_nat (i := 3) F.c10.h323_0 (by norm_num)
  have hkeys : ![![generatedRoot.w_320_0 ctx, generatedRoot.w_324_0 ctx],
      ![generatedRoot.w_321_0 ctx, generatedRoot.w_325_0 ctx]] = keyPairG hashModel
        (generatedRoot.w_7_0 ctx) modulus_0 25 helperPre [.integer (sigmaOf (W := 4) sigmas4 ⟨3, by omega⟩)] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact keyPairG_of_hash (pre := [.integer (sigmaOf (W := 4) sigmas4 ⟨3, by omega⟩)]) 0 0 (by norm_num) F.c10.h320_0
    · exact keyPairG_of_hash (pre := [.integer (sigmaOf (W := 4) sigmas4 ⟨3, by omega⟩)]) 0 1 (by norm_num) F.c10.h324_0
    · exact keyPairG_of_hash (pre := [.integer (sigmaOf (W := 4) sigmas4 ⟨3, by omega⟩)]) 1 0 (by norm_num) F.c10.h321_0
    · exact keyPairG_of_hash (pre := [.integer (sigmaOf (W := 4) sigmas4 ⟨3, by omega⟩)]) 1 1 (by norm_num) F.c10.h325_0
  have hhelpL := hhelp ⟨3, by omega⟩
  rw [← hkeys] at hhelpL
  have hh0 : concatRows (![generatedRoot.w_255_0 ctx ⟨3, by omega⟩, generatedRoot.w_256_0 ctx ⟨3, by omega⟩] 0)
      (![![generatedRoot.w_320_0 ctx, generatedRoot.w_324_0 ctx],
        ![generatedRoot.w_321_0 ctx, generatedRoot.w_325_0 ctx]] 0 0) (generatedRoot.w_322_concat_1 ctx) := by
    have h := F.c10.h320_0_1; rw [e318] at h; exact h
  have hh1 : concatRows (![generatedRoot.w_255_0 ctx ⟨3, by omega⟩, generatedRoot.w_256_0 ctx ⟨3, by omega⟩] 1)
      (![![generatedRoot.w_320_0 ctx, generatedRoot.w_324_0 ctx],
        ![generatedRoot.w_321_0 ctx, generatedRoot.w_325_0 ctx]] 0 1) (generatedRoot.w_326_concat_1 ctx) := by
    have h := F.c10.h324_0_1; rw [e323] at h; exact h
  refine ⟨?_, ?_⟩
  · exact lut_round_flat_at (by norm_num) 0 (log := 3) (by norm_num) (L := 3) (W := 4) rfl hq1
      layout hlayout hexact (sigmaOf sigmas4) sigmas4_odd hsm hg' hg henc (fam := fam) hrel
      (h3 := ![generatedRoot.w_255_0 ctx ⟨3, by omega⟩, generatedRoot.w_256_0 ctx ⟨3, by omega⟩]) hhelpL
      (rc0 := generatedRoot.w_328_0 ctx) (rA00 := generatedRoot.w_329_0 ctx)
      (rA10 := generatedRoot.w_330_0 ctx) (rc1 := generatedRoot.w_332_0 ctx)
      (rA01 := generatedRoot.w_333_0 ctx) (rA11 := generatedRoot.w_334_0 ctx)
      F.c11.h328_0 F.c11.h329_0 F.c11.h330_0 F.c11.h332_0 F.c11.h333_0 F.c11.h334_0
      F.c11.h329_0_1 F.c11.h331_0 F.c11.h333_0_1 F.c11.h335_0 F.c11.h336_0
      (by show concatRows (fam 0 0 _) (fam 1 0 _) _; rw [← hs00, ← hs10]; exact F.c11.h337_0)
      F.c11.h338_0 F.c11.h339_0 F.c11.h341_0 F.c12.h342_0 F.c12.h343_0 F.c12.h344_0 F.c12.h345_0
      hh0 F.c10.h322_0 hh1 F.c10.h326_0 F.c10.h327_0 F.c12.h347_0 F.c12.h348_0 (hswitch 0) hμ hwalsh
  · exact lut_round_flat_at (by norm_num) 1 (log := 3) (by norm_num) (L := 3) (W := 4) rfl hq1
      layout hlayout hexact (sigmaOf sigmas4) sigmas4_odd hsm hg' hg henc (fam := fam) hrel
      (h3 := ![generatedRoot.w_255_0 ctx ⟨3, by omega⟩, generatedRoot.w_256_0 ctx ⟨3, by omega⟩]) hhelpL
      (rc0 := generatedRoot.w_328_0 ctx) (rA00 := generatedRoot.w_329_0 ctx)
      (rA10 := generatedRoot.w_330_0 ctx) (rc1 := generatedRoot.w_332_0 ctx)
      (rA01 := generatedRoot.w_333_0 ctx) (rA11 := generatedRoot.w_334_0 ctx)
      F.c11.h328_0 F.c11.h329_0 F.c11.h330_0 F.c11.h332_0 F.c11.h333_0 F.c11.h334_0
      F.c11.h329_0_1 F.c11.h331_0 F.c11.h333_0_1 F.c11.h335_0 F.c11.h336_0
      (by show concatRows (fam 0 1 _) (fam 1 1 _) _; rw [← hs01, ← hs11]; exact F.c12.h357_0)
      F.c12.h358_0 F.c12.h359_0 F.c13.h361_0 F.c13.h362_0 F.c13.h363_0 F.c13.h364_0 F.c13.h365_0
      hh0 F.c10.h322_0 hh1 F.c10.h326_0 F.c10.h327_0 F.c13.h367_0 F.c13.h368_0 (hswitch 1) hμ hwalsh

end Dio2Proof
