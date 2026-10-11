import Part_0_1_SmudgeTotal
import Part_0_1_RootSums

/-! The smudge carriers of part `0_1` over the root wires (written by
`generator/gen_smudge_chain.py`). -/

namespace RefreshTopProof.Part_0_1

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof


set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The smudge carrier of decoder `0`. -/
theorem smudgeChain0 {model : HashModel} {params : W_Stage_online_eval_0_1.Params}
    {w_532_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_533_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_534_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_535_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_536_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_537_0 : ByteArray}
    {w_538_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_539_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_540_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_541_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_542_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_543_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_544_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_545_0 : ExactMatrix Q6 8 1 10}
    {w_546_0 : ExactMatrix Q6 8 1 10}
    {w_547_0 : ExactMatrix Q6 8 1 10}
    {w_549_0 : ExactMatrix Q6 8 1 10}
    {w_550_0 : ExactMatrix Q6 8 1 10}
    {w_551_0 : ExactMatrix Q6 8 1 10}
    {w_552_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_553_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_556_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_557_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_548_0 : Fin 5 → ExactMatrix Q6 8 1 1}
    {w_554_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_554_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_554_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_554_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_554_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_554_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_555_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_555_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_555_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_555_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_555_4 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_555_5 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_558_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_558_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_558_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_558_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_558_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_558_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_559_0 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_559_1 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_559_2 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_559_3 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_559_4 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_559_5 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_560_0 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_560_1 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_560_2 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_560_3 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_560_4 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_560_5 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_561_0 : ExactMatrix Q1 8 1 1}
    {w_562_0 : ExactMatrix Q1 8 1 1}
    {w_563_0 : ExactMatrix Q1 8 1 1}
    {w_564_0 : ExactMatrix Q1 8 1 1}
    {w_565_0 : ExactMatrix Q1 8 1 1}
    {w_566_0 : ExactMatrix Q1 8 1 1}
    {w_567_0 : ExactMatrix Q1 8 1 1}
    {w_568_0 : ExactMatrix Q1 8 1 1}
    {w_569_0 : ExactMatrix Q1 8 1 1}
    {w_570_0 : ExactMatrix Q1 8 1 1}
    {w_571_0 : ExactMatrix Q1 8 1 1}
    {w_572_0 : ExactMatrix Q1 8 1 1}
    {w_573_0 : ExactMatrix Q1 8 1 1}
    {w_574_0 : ExactMatrix Q1 8 1 1}
    {w_575_0 : ExactMatrix Q1 8 1 1}
    {w_576_0 : ExactMatrix Q1 8 1 1}
    {w_577_0 : ExactMatrix Q1 8 1 1}
    {w_578_0 : ExactMatrix Q1 8 1 1}
    {w_579_0 : ExactMatrix Q1 8 1 1}
    {w_579_1 : ExactMatrix Q1 8 1 1}
    {w_579_2 : ExactMatrix Q1 8 1 1}
    {w_579_3 : ExactMatrix Q1 8 1 1}
    {w_579_4 : ExactMatrix Q1 8 1 1}
    {w_579_5 : ExactMatrix Q1 8 1 1}
    (e_563_0 : w_563_0 = matrixAdd w_561_0 w_562_0)
    (e_566_0 : w_566_0 = matrixAdd w_564_0 w_565_0)
    (e_569_0 : w_569_0 = matrixAdd w_567_0 w_568_0)
    (e_572_0 : w_572_0 = matrixAdd w_570_0 w_571_0)
    (e_575_0 : w_575_0 = matrixAdd w_573_0 w_574_0)
    (e_578_0 : w_578_0 = matrixAdd w_576_0 w_577_0)
    (h554_5 : (∀ i : Fin 800, W_Stage_online_eval_0_1.parallel_generatedRoot_554 Backend.backend model params i (w_539_0, (w_540_0, (w_541_0, (w_542_0, (w_543_0, (w_544_0, (w_545_0, (w_546_0, (w_547_0, (w_548_0, (w_549_0, (w_550_0, (w_551_0, (w_537_0, ((w_552_0 i), ((w_553_0 i), ())))))))))))))))) ((w_554_0 i), ((w_554_1 i), ((w_554_2 i), ((w_554_3 i), ((w_554_4 i), ((w_554_5 i), ()))))))))
    (h555_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_555 Backend.backend model params i (w_554_0, (w_554_1, (w_554_2, (w_554_3, (w_554_4, (w_554_5, ())))))) ((w_555_0 i), ((w_555_1 i), ((w_555_2 i), ((w_555_3 i), ((w_555_4 i), ((w_555_5 i), ()))))))))
    (h558_5 : (∀ i : Fin 800, W_Stage_online_eval_0_1.parallel_generatedRoot_558 Backend.backend model params i (w_539_0, (w_540_0, (w_541_0, (w_542_0, (w_543_0, (w_544_0, (w_555_0, (w_555_2, (w_555_4, (w_548_0, (w_555_1, (w_555_3, (w_555_5, (w_537_0, ((w_556_0 i), ((w_557_0 i), ())))))))))))))))) ((w_558_0 i), ((w_558_1 i), ((w_558_2 i), ((w_558_3 i), ((w_558_4 i), ((w_558_5 i), ()))))))))
    (h559_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_559 Backend.backend model params i (w_558_0, (w_558_1, (w_558_2, (w_558_3, (w_558_4, (w_558_5, ())))))) ((w_559_0 i), ((w_559_1 i), ((w_559_2 i), ((w_559_3 i), ((w_559_4 i), ((w_559_5 i), ()))))))))
    (h560_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_560 Backend.backend model params i (w_532_0, (w_533_0, (w_534_0, (w_535_0, (w_536_0, (w_537_0, (w_538_0, ((w_559_4 i), ((w_559_0 i), ((w_559_2 i), ((w_559_5 i), ((w_559_1 i), ((w_559_3 i), ()))))))))))))) ((w_560_0 i), ((w_560_1 i), ((w_560_2 i), ((w_560_3 i), ((w_560_4 i), ((w_560_5 i), ()))))))))
    (h561_0 : familyGetStatic w_560_0 (0) w_561_0)
    (h562_0 : familyGetStatic w_560_0 (1) w_562_0)
    (h564_0 : familyGetStatic w_560_1 (0) w_564_0)
    (h565_0 : familyGetStatic w_560_1 (1) w_565_0)
    (h567_0 : familyGetStatic w_560_2 (0) w_567_0)
    (h568_0 : familyGetStatic w_560_2 (1) w_568_0)
    (h570_0 : familyGetStatic w_560_3 (0) w_570_0)
    (h571_0 : familyGetStatic w_560_3 (1) w_571_0)
    (h573_0 : familyGetStatic w_560_4 (0) w_573_0)
    (h574_0 : familyGetStatic w_560_4 (1) w_574_0)
    (h576_0 : familyGetStatic w_560_5 (0) w_576_0)
    (h577_0 : familyGetStatic w_560_5 (1) w_577_0)
    (h579_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => W_Stage_online_eval_0_1.sequential_generatedRoot_579 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_560_0, (w_560_1, (w_560_2, (w_560_3, (w_560_4, (w_560_5, ())))))))))))) next) (Int.toNat (158)) (w_563_0, (w_566_0, (w_569_0, (w_572_0, (w_575_0, (w_578_0, ())))))) (w_579_0, (w_579_1, (w_579_2, (w_579_3, (w_579_4, (w_579_5, ())))))))
    {sm sp : ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12}
    (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10}
    {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6')
    (hg7 : centeredRebaseRuns g6' g7)
    {g5' : ExactMatrix Q0 N 1 8}
    {h6 : ExactMatrix Q5 N 1 8}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 8 g5')
    (hh6 : centeredRebaseRuns g5' h6)
    {h5 : ExactMatrix Q0 N 1 6}
    {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![w_540_0 k, w_543_0 k], ![w_541_0 k, w_544_0 k]] (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![w_539_0 k, w_542_0 k] Ba)
    (hpay : ∀ k : Fin 5, w_548_0 k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat}
    (hone : Rel sm ![![w_545_0, w_549_0], ![w_546_0, w_550_0]] (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![w_547_0, w_551_0] Bs)
    {s : Fin 800 → Int}
    {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8}
    {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model w_537_0 Q5 8 smudge0Rhs0 i j + ![w_552_0 i, w_553_0 i] j = reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 + reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int}
    {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6}
    {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model w_537_0 Q0 6 smudge0Rhs1 i j + ![w_556_0 i, w_557_0 i] j = reducePoly Q0 N (((q3 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 + reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {g1' : ExactMatrix Q1 N 1 6}
    (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1')
    (hh5 : centeredRebaseRuns g1' h5)
    {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model w_537_0 Q0 6 2 (sigma i)) (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) * unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![w_536_0 i, w_538_0 i] Bh)
    {gq : ExactMatrix Q1 N 1 6}
    (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq)
    {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1)
    {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j, reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) • ![![w_532_0, w_533_0], ![w_534_0, w_535_0]] 0 j i + ![![w_532_0, w_533_0], ![w_534_0, w_535_0]] 1 j i = (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (smudgeTable k)) (sigma i).toNat * reducePoly Q1 N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    Rel sm ![![w_579_0, w_579_1], ![w_579_2, w_579_3]]
      (fun j ↦ reducePoly Q1 N (smudgeValue0 (fun o ↦ a (sel' o) + a (sel o)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![w_579_4, w_579_5] (160 * smudge0LutBound Ba Bs BB BB' Bh BS) := by
  subst e_563_0 e_566_0 e_569_0 e_572_0 e_575_0 e_578_0
  have hs := rootSum579 h561_0 h562_0 h564_0 h565_0 h567_0 h568_0 h570_0 h571_0 h573_0 h574_0 h576_0 h577_0 h579_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  rw [s0, s1, s2, s3, s4, s5]
  exact smudgeTotal0 (model := model) (h554 := h554_5) (h555 := h555_5) (h558 := h558_5) (h559 := h559_5) (hlut := h560_5) (hsm := hsm) (hG := hG) (hg6' := hg6') (hg7 := hg7) (hg5' := hg5') (hh6 := hh6) (ha := ha) (hpay := hpay) (hone := hone) (hrhs := hrhs) (heB := heB) (hrhs' := hrhs') (heB' := heB') (hsel := hsel) (hsel' := hsel') (hg1' := hg1') (hh5 := hh5) (hhelp := hhelp) (hgq := hgq) (hinverse := hinverse) (hswitch := hswitch)


set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The smudge carrier of decoder `1`. -/
theorem smudgeChain1 {model : HashModel} {params : W_Stage_online_eval_0_1.Params}
    {w_587_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_588_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_589_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_590_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_591_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_592_0 : ByteArray}
    {w_593_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_594_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_595_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_596_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_597_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_598_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_599_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_600_0 : ExactMatrix Q6 8 1 10}
    {w_601_0 : ExactMatrix Q6 8 1 10}
    {w_602_0 : ExactMatrix Q6 8 1 10}
    {w_604_0 : ExactMatrix Q6 8 1 10}
    {w_605_0 : ExactMatrix Q6 8 1 10}
    {w_606_0 : ExactMatrix Q6 8 1 10}
    {w_607_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_608_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_611_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_612_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_603_0 : Fin 5 → ExactMatrix Q6 8 1 1}
    {w_609_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_609_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_609_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_609_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_609_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_609_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_610_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_610_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_610_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_610_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_610_4 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_610_5 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_613_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_613_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_613_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_613_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_613_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_613_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_614_0 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_614_1 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_614_2 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_614_3 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_614_4 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_614_5 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_615_0 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_615_1 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_615_2 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_615_3 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_615_4 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_615_5 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_616_0 : ExactMatrix Q1 8 1 1}
    {w_617_0 : ExactMatrix Q1 8 1 1}
    {w_618_0 : ExactMatrix Q1 8 1 1}
    {w_619_0 : ExactMatrix Q1 8 1 1}
    {w_620_0 : ExactMatrix Q1 8 1 1}
    {w_621_0 : ExactMatrix Q1 8 1 1}
    {w_622_0 : ExactMatrix Q1 8 1 1}
    {w_623_0 : ExactMatrix Q1 8 1 1}
    {w_624_0 : ExactMatrix Q1 8 1 1}
    {w_625_0 : ExactMatrix Q1 8 1 1}
    {w_626_0 : ExactMatrix Q1 8 1 1}
    {w_627_0 : ExactMatrix Q1 8 1 1}
    {w_628_0 : ExactMatrix Q1 8 1 1}
    {w_629_0 : ExactMatrix Q1 8 1 1}
    {w_630_0 : ExactMatrix Q1 8 1 1}
    {w_631_0 : ExactMatrix Q1 8 1 1}
    {w_632_0 : ExactMatrix Q1 8 1 1}
    {w_633_0 : ExactMatrix Q1 8 1 1}
    {w_634_0 : ExactMatrix Q1 8 1 1}
    {w_634_1 : ExactMatrix Q1 8 1 1}
    {w_634_2 : ExactMatrix Q1 8 1 1}
    {w_634_3 : ExactMatrix Q1 8 1 1}
    {w_634_4 : ExactMatrix Q1 8 1 1}
    {w_634_5 : ExactMatrix Q1 8 1 1}
    (e_618_0 : w_618_0 = matrixAdd w_616_0 w_617_0)
    (e_621_0 : w_621_0 = matrixAdd w_619_0 w_620_0)
    (e_624_0 : w_624_0 = matrixAdd w_622_0 w_623_0)
    (e_627_0 : w_627_0 = matrixAdd w_625_0 w_626_0)
    (e_630_0 : w_630_0 = matrixAdd w_628_0 w_629_0)
    (e_633_0 : w_633_0 = matrixAdd w_631_0 w_632_0)
    (h609_5 : (∀ i : Fin 800, W_Stage_online_eval_0_1.parallel_generatedRoot_609 Backend.backend model params i (w_594_0, (w_595_0, (w_596_0, (w_597_0, (w_598_0, (w_599_0, (w_600_0, (w_601_0, (w_602_0, (w_603_0, (w_604_0, (w_605_0, (w_606_0, (w_592_0, ((w_607_0 i), ((w_608_0 i), ())))))))))))))))) ((w_609_0 i), ((w_609_1 i), ((w_609_2 i), ((w_609_3 i), ((w_609_4 i), ((w_609_5 i), ()))))))))
    (h610_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_610 Backend.backend model params i (w_609_0, (w_609_1, (w_609_2, (w_609_3, (w_609_4, (w_609_5, ())))))) ((w_610_0 i), ((w_610_1 i), ((w_610_2 i), ((w_610_3 i), ((w_610_4 i), ((w_610_5 i), ()))))))))
    (h613_5 : (∀ i : Fin 800, W_Stage_online_eval_0_1.parallel_generatedRoot_613 Backend.backend model params i (w_594_0, (w_595_0, (w_596_0, (w_597_0, (w_598_0, (w_599_0, (w_610_0, (w_610_2, (w_610_4, (w_603_0, (w_610_1, (w_610_3, (w_610_5, (w_592_0, ((w_611_0 i), ((w_612_0 i), ())))))))))))))))) ((w_613_0 i), ((w_613_1 i), ((w_613_2 i), ((w_613_3 i), ((w_613_4 i), ((w_613_5 i), ()))))))))
    (h614_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_614 Backend.backend model params i (w_613_0, (w_613_1, (w_613_2, (w_613_3, (w_613_4, (w_613_5, ())))))) ((w_614_0 i), ((w_614_1 i), ((w_614_2 i), ((w_614_3 i), ((w_614_4 i), ((w_614_5 i), ()))))))))
    (h615_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_615 Backend.backend model params i (w_587_0, (w_588_0, (w_589_0, (w_590_0, (w_591_0, (w_592_0, (w_593_0, ((w_614_4 i), ((w_614_0 i), ((w_614_2 i), ((w_614_5 i), ((w_614_1 i), ((w_614_3 i), ()))))))))))))) ((w_615_0 i), ((w_615_1 i), ((w_615_2 i), ((w_615_3 i), ((w_615_4 i), ((w_615_5 i), ()))))))))
    (h616_0 : familyGetStatic w_615_0 (0) w_616_0)
    (h617_0 : familyGetStatic w_615_0 (1) w_617_0)
    (h619_0 : familyGetStatic w_615_1 (0) w_619_0)
    (h620_0 : familyGetStatic w_615_1 (1) w_620_0)
    (h622_0 : familyGetStatic w_615_2 (0) w_622_0)
    (h623_0 : familyGetStatic w_615_2 (1) w_623_0)
    (h625_0 : familyGetStatic w_615_3 (0) w_625_0)
    (h626_0 : familyGetStatic w_615_3 (1) w_626_0)
    (h628_0 : familyGetStatic w_615_4 (0) w_628_0)
    (h629_0 : familyGetStatic w_615_4 (1) w_629_0)
    (h631_0 : familyGetStatic w_615_5 (0) w_631_0)
    (h632_0 : familyGetStatic w_615_5 (1) w_632_0)
    (h634_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => W_Stage_online_eval_0_1.sequential_generatedRoot_634 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_615_0, (w_615_1, (w_615_2, (w_615_3, (w_615_4, (w_615_5, ())))))))))))) next) (Int.toNat (158)) (w_618_0, (w_621_0, (w_624_0, (w_627_0, (w_630_0, (w_633_0, ())))))) (w_634_0, (w_634_1, (w_634_2, (w_634_3, (w_634_4, (w_634_5, ())))))))
    {sm sp : ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12}
    (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10}
    {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6')
    (hg7 : centeredRebaseRuns g6' g7)
    {g5' : ExactMatrix Q0 N 1 8}
    {h6 : ExactMatrix Q5 N 1 8}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 8 g5')
    (hh6 : centeredRebaseRuns g5' h6)
    {h5 : ExactMatrix Q0 N 1 6}
    {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![w_595_0 k, w_598_0 k], ![w_596_0 k, w_599_0 k]] (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![w_594_0 k, w_597_0 k] Ba)
    (hpay : ∀ k : Fin 5, w_603_0 k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat}
    (hone : Rel sm ![![w_600_0, w_604_0], ![w_601_0, w_605_0]] (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![w_602_0, w_606_0] Bs)
    {s : Fin 800 → Int}
    {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8}
    {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model w_592_0 Q5 8 smudge1Rhs0 i j + ![w_607_0 i, w_608_0 i] j = reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 + reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int}
    {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6}
    {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model w_592_0 Q0 6 smudge1Rhs1 i j + ![w_611_0 i, w_612_0 i] j = reducePoly Q0 N (((q3 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 + reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {g1' : ExactMatrix Q1 N 1 6}
    (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1')
    (hh5 : centeredRebaseRuns g1' h5)
    {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model w_592_0 Q0 6 2 (sigma i)) (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) * unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![w_591_0 i, w_593_0 i] Bh)
    {gq : ExactMatrix Q1 N 1 6}
    (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq)
    {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1)
    {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j, reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) • ![![w_587_0, w_588_0], ![w_589_0, w_590_0]] 0 j i + ![![w_587_0, w_588_0], ![w_589_0, w_590_0]] 1 j i = (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (smudgeTable k)) (sigma i).toNat * reducePoly Q1 N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    Rel sm ![![w_634_0, w_634_1], ![w_634_2, w_634_3]]
      (fun j ↦ reducePoly Q1 N (smudgeValue1 (fun o ↦ a (sel' o) + a (sel o)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![w_634_4, w_634_5] (160 * smudge1LutBound Ba Bs BB BB' Bh BS) := by
  subst e_618_0 e_621_0 e_624_0 e_627_0 e_630_0 e_633_0
  have hs := rootSum634 h616_0 h617_0 h619_0 h620_0 h622_0 h623_0 h625_0 h626_0 h628_0 h629_0 h631_0 h632_0 h634_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  rw [s0, s1, s2, s3, s4, s5]
  exact smudgeTotal1 (model := model) (h609 := h609_5) (h610 := h610_5) (h613 := h613_5) (h614 := h614_5) (hlut := h615_5) (hsm := hsm) (hG := hG) (hg6' := hg6') (hg7 := hg7) (hg5' := hg5') (hh6 := hh6) (ha := ha) (hpay := hpay) (hone := hone) (hrhs := hrhs) (heB := heB) (hrhs' := hrhs') (heB' := heB') (hsel := hsel) (hsel' := hsel') (hg1' := hg1') (hh5 := hh5) (hhelp := hhelp) (hgq := hgq) (hinverse := hinverse) (hswitch := hswitch)


set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The smudge carrier of decoder `2`. -/
theorem smudgeChain2 {model : HashModel} {params : W_Stage_online_eval_0_1.Params}
    {w_641_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_642_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_643_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_644_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_645_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_646_0 : ByteArray}
    {w_647_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_648_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_649_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_650_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_651_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_652_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_653_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_654_0 : ExactMatrix Q6 8 1 10}
    {w_655_0 : ExactMatrix Q6 8 1 10}
    {w_656_0 : ExactMatrix Q6 8 1 10}
    {w_658_0 : ExactMatrix Q6 8 1 10}
    {w_659_0 : ExactMatrix Q6 8 1 10}
    {w_660_0 : ExactMatrix Q6 8 1 10}
    {w_661_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_662_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_665_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_666_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_657_0 : Fin 5 → ExactMatrix Q6 8 1 1}
    {w_663_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_663_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_663_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_663_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_663_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_663_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_664_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_664_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_664_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_664_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_664_4 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_664_5 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_667_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_667_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_667_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_667_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_667_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_667_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_668_0 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_668_1 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_668_2 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_668_3 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_668_4 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_668_5 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_669_0 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_669_1 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_669_2 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_669_3 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_669_4 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_669_5 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_670_0 : ExactMatrix Q1 8 1 1}
    {w_671_0 : ExactMatrix Q1 8 1 1}
    {w_672_0 : ExactMatrix Q1 8 1 1}
    {w_673_0 : ExactMatrix Q1 8 1 1}
    {w_674_0 : ExactMatrix Q1 8 1 1}
    {w_675_0 : ExactMatrix Q1 8 1 1}
    {w_676_0 : ExactMatrix Q1 8 1 1}
    {w_677_0 : ExactMatrix Q1 8 1 1}
    {w_678_0 : ExactMatrix Q1 8 1 1}
    {w_679_0 : ExactMatrix Q1 8 1 1}
    {w_680_0 : ExactMatrix Q1 8 1 1}
    {w_681_0 : ExactMatrix Q1 8 1 1}
    {w_682_0 : ExactMatrix Q1 8 1 1}
    {w_683_0 : ExactMatrix Q1 8 1 1}
    {w_684_0 : ExactMatrix Q1 8 1 1}
    {w_685_0 : ExactMatrix Q1 8 1 1}
    {w_686_0 : ExactMatrix Q1 8 1 1}
    {w_687_0 : ExactMatrix Q1 8 1 1}
    {w_688_0 : ExactMatrix Q1 8 1 1}
    {w_688_1 : ExactMatrix Q1 8 1 1}
    {w_688_2 : ExactMatrix Q1 8 1 1}
    {w_688_3 : ExactMatrix Q1 8 1 1}
    {w_688_4 : ExactMatrix Q1 8 1 1}
    {w_688_5 : ExactMatrix Q1 8 1 1}
    (e_672_0 : w_672_0 = matrixAdd w_670_0 w_671_0)
    (e_675_0 : w_675_0 = matrixAdd w_673_0 w_674_0)
    (e_678_0 : w_678_0 = matrixAdd w_676_0 w_677_0)
    (e_681_0 : w_681_0 = matrixAdd w_679_0 w_680_0)
    (e_684_0 : w_684_0 = matrixAdd w_682_0 w_683_0)
    (e_687_0 : w_687_0 = matrixAdd w_685_0 w_686_0)
    (h663_5 : (∀ i : Fin 800, W_Stage_online_eval_0_1.parallel_generatedRoot_663 Backend.backend model params i (w_648_0, (w_649_0, (w_650_0, (w_651_0, (w_652_0, (w_653_0, (w_654_0, (w_655_0, (w_656_0, (w_657_0, (w_658_0, (w_659_0, (w_660_0, (w_646_0, ((w_661_0 i), ((w_662_0 i), ())))))))))))))))) ((w_663_0 i), ((w_663_1 i), ((w_663_2 i), ((w_663_3 i), ((w_663_4 i), ((w_663_5 i), ()))))))))
    (h664_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_664 Backend.backend model params i (w_663_0, (w_663_1, (w_663_2, (w_663_3, (w_663_4, (w_663_5, ())))))) ((w_664_0 i), ((w_664_1 i), ((w_664_2 i), ((w_664_3 i), ((w_664_4 i), ((w_664_5 i), ()))))))))
    (h667_5 : (∀ i : Fin 800, W_Stage_online_eval_0_1.parallel_generatedRoot_667 Backend.backend model params i (w_648_0, (w_649_0, (w_650_0, (w_651_0, (w_652_0, (w_653_0, (w_664_0, (w_664_2, (w_664_4, (w_657_0, (w_664_1, (w_664_3, (w_664_5, (w_646_0, ((w_665_0 i), ((w_666_0 i), ())))))))))))))))) ((w_667_0 i), ((w_667_1 i), ((w_667_2 i), ((w_667_3 i), ((w_667_4 i), ((w_667_5 i), ()))))))))
    (h668_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_668 Backend.backend model params i (w_667_0, (w_667_1, (w_667_2, (w_667_3, (w_667_4, (w_667_5, ())))))) ((w_668_0 i), ((w_668_1 i), ((w_668_2 i), ((w_668_3 i), ((w_668_4 i), ((w_668_5 i), ()))))))))
    (h669_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_669 Backend.backend model params i (w_641_0, (w_642_0, (w_643_0, (w_644_0, (w_645_0, (w_646_0, (w_647_0, ((w_668_4 i), ((w_668_0 i), ((w_668_2 i), ((w_668_5 i), ((w_668_1 i), ((w_668_3 i), ()))))))))))))) ((w_669_0 i), ((w_669_1 i), ((w_669_2 i), ((w_669_3 i), ((w_669_4 i), ((w_669_5 i), ()))))))))
    (h670_0 : familyGetStatic w_669_0 (0) w_670_0)
    (h671_0 : familyGetStatic w_669_0 (1) w_671_0)
    (h673_0 : familyGetStatic w_669_1 (0) w_673_0)
    (h674_0 : familyGetStatic w_669_1 (1) w_674_0)
    (h676_0 : familyGetStatic w_669_2 (0) w_676_0)
    (h677_0 : familyGetStatic w_669_2 (1) w_677_0)
    (h679_0 : familyGetStatic w_669_3 (0) w_679_0)
    (h680_0 : familyGetStatic w_669_3 (1) w_680_0)
    (h682_0 : familyGetStatic w_669_4 (0) w_682_0)
    (h683_0 : familyGetStatic w_669_4 (1) w_683_0)
    (h685_0 : familyGetStatic w_669_5 (0) w_685_0)
    (h686_0 : familyGetStatic w_669_5 (1) w_686_0)
    (h688_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => W_Stage_online_eval_0_1.sequential_generatedRoot_688 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_669_0, (w_669_1, (w_669_2, (w_669_3, (w_669_4, (w_669_5, ())))))))))))) next) (Int.toNat (158)) (w_672_0, (w_675_0, (w_678_0, (w_681_0, (w_684_0, (w_687_0, ())))))) (w_688_0, (w_688_1, (w_688_2, (w_688_3, (w_688_4, (w_688_5, ())))))))
    {sm sp : ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12}
    (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10}
    {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6')
    (hg7 : centeredRebaseRuns g6' g7)
    {g5' : ExactMatrix Q0 N 1 8}
    {h6 : ExactMatrix Q5 N 1 8}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 8 g5')
    (hh6 : centeredRebaseRuns g5' h6)
    {h5 : ExactMatrix Q0 N 1 6}
    {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![w_649_0 k, w_652_0 k], ![w_650_0 k, w_653_0 k]] (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![w_648_0 k, w_651_0 k] Ba)
    (hpay : ∀ k : Fin 5, w_657_0 k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat}
    (hone : Rel sm ![![w_654_0, w_658_0], ![w_655_0, w_659_0]] (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![w_656_0, w_660_0] Bs)
    {s : Fin 800 → Int}
    {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8}
    {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model w_646_0 Q5 8 smudge2Rhs0 i j + ![w_661_0 i, w_662_0 i] j = reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 + reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int}
    {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6}
    {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model w_646_0 Q0 6 smudge2Rhs1 i j + ![w_665_0 i, w_666_0 i] j = reducePoly Q0 N (((q3 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 + reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {g1' : ExactMatrix Q1 N 1 6}
    (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1')
    (hh5 : centeredRebaseRuns g1' h5)
    {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model w_646_0 Q0 6 2 (sigma i)) (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) * unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![w_645_0 i, w_647_0 i] Bh)
    {gq : ExactMatrix Q1 N 1 6}
    (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq)
    {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1)
    {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j, reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) • ![![w_641_0, w_642_0], ![w_643_0, w_644_0]] 0 j i + ![![w_641_0, w_642_0], ![w_643_0, w_644_0]] 1 j i = (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (smudgeTable k)) (sigma i).toNat * reducePoly Q1 N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    Rel sm ![![w_688_0, w_688_1], ![w_688_2, w_688_3]]
      (fun j ↦ reducePoly Q1 N (smudgeValue2 (fun o ↦ a (sel' o) + a (sel o)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![w_688_4, w_688_5] (160 * smudge2LutBound Ba Bs BB BB' Bh BS) := by
  subst e_672_0 e_675_0 e_678_0 e_681_0 e_684_0 e_687_0
  have hs := rootSum688 h670_0 h671_0 h673_0 h674_0 h676_0 h677_0 h679_0 h680_0 h682_0 h683_0 h685_0 h686_0 h688_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  rw [s0, s1, s2, s3, s4, s5]
  exact smudgeTotal2 (model := model) (h663 := h663_5) (h664 := h664_5) (h667 := h667_5) (h668 := h668_5) (hlut := h669_5) (hsm := hsm) (hG := hG) (hg6' := hg6') (hg7 := hg7) (hg5' := hg5') (hh6 := hh6) (ha := ha) (hpay := hpay) (hone := hone) (hrhs := hrhs) (heB := heB) (hrhs' := hrhs') (heB' := heB') (hsel := hsel) (hsel' := hsel') (hg1' := hg1') (hh5 := hh5) (hhelp := hhelp) (hgq := hgq) (hinverse := hinverse) (hswitch := hswitch)


set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The smudge carrier of decoder `3`. -/
theorem smudgeChain3 {model : HashModel} {params : W_Stage_online_eval_0_1.Params}
    {w_695_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_696_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_697_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_698_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_699_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_700_0 : ByteArray}
    {w_701_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_702_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_703_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_704_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_705_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_706_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_707_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_708_0 : ExactMatrix Q6 8 1 10}
    {w_709_0 : ExactMatrix Q6 8 1 10}
    {w_710_0 : ExactMatrix Q6 8 1 10}
    {w_712_0 : ExactMatrix Q6 8 1 10}
    {w_713_0 : ExactMatrix Q6 8 1 10}
    {w_714_0 : ExactMatrix Q6 8 1 10}
    {w_715_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_716_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_719_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_720_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_711_0 : Fin 5 → ExactMatrix Q6 8 1 1}
    {w_717_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_717_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_717_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_717_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_717_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_717_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_718_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_718_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_718_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_718_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_718_4 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_718_5 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_721_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_721_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_721_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_721_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_721_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_721_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_722_0 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_722_1 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_722_2 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_722_3 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_722_4 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_722_5 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_723_0 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_723_1 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_723_2 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_723_3 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_723_4 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_723_5 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_724_0 : ExactMatrix Q1 8 1 1}
    {w_725_0 : ExactMatrix Q1 8 1 1}
    {w_726_0 : ExactMatrix Q1 8 1 1}
    {w_727_0 : ExactMatrix Q1 8 1 1}
    {w_728_0 : ExactMatrix Q1 8 1 1}
    {w_729_0 : ExactMatrix Q1 8 1 1}
    {w_730_0 : ExactMatrix Q1 8 1 1}
    {w_731_0 : ExactMatrix Q1 8 1 1}
    {w_732_0 : ExactMatrix Q1 8 1 1}
    {w_733_0 : ExactMatrix Q1 8 1 1}
    {w_734_0 : ExactMatrix Q1 8 1 1}
    {w_735_0 : ExactMatrix Q1 8 1 1}
    {w_736_0 : ExactMatrix Q1 8 1 1}
    {w_737_0 : ExactMatrix Q1 8 1 1}
    {w_738_0 : ExactMatrix Q1 8 1 1}
    {w_739_0 : ExactMatrix Q1 8 1 1}
    {w_740_0 : ExactMatrix Q1 8 1 1}
    {w_741_0 : ExactMatrix Q1 8 1 1}
    {w_742_0 : ExactMatrix Q1 8 1 1}
    {w_742_1 : ExactMatrix Q1 8 1 1}
    {w_742_2 : ExactMatrix Q1 8 1 1}
    {w_742_3 : ExactMatrix Q1 8 1 1}
    {w_742_4 : ExactMatrix Q1 8 1 1}
    {w_742_5 : ExactMatrix Q1 8 1 1}
    (e_726_0 : w_726_0 = matrixAdd w_724_0 w_725_0)
    (e_729_0 : w_729_0 = matrixAdd w_727_0 w_728_0)
    (e_732_0 : w_732_0 = matrixAdd w_730_0 w_731_0)
    (e_735_0 : w_735_0 = matrixAdd w_733_0 w_734_0)
    (e_738_0 : w_738_0 = matrixAdd w_736_0 w_737_0)
    (e_741_0 : w_741_0 = matrixAdd w_739_0 w_740_0)
    (h717_5 : (∀ i : Fin 800, W_Stage_online_eval_0_1.parallel_generatedRoot_717 Backend.backend model params i (w_702_0, (w_703_0, (w_704_0, (w_705_0, (w_706_0, (w_707_0, (w_708_0, (w_709_0, (w_710_0, (w_711_0, (w_712_0, (w_713_0, (w_714_0, (w_700_0, ((w_715_0 i), ((w_716_0 i), ())))))))))))))))) ((w_717_0 i), ((w_717_1 i), ((w_717_2 i), ((w_717_3 i), ((w_717_4 i), ((w_717_5 i), ()))))))))
    (h718_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_718 Backend.backend model params i (w_717_0, (w_717_1, (w_717_2, (w_717_3, (w_717_4, (w_717_5, ())))))) ((w_718_0 i), ((w_718_1 i), ((w_718_2 i), ((w_718_3 i), ((w_718_4 i), ((w_718_5 i), ()))))))))
    (h721_5 : (∀ i : Fin 800, W_Stage_online_eval_0_1.parallel_generatedRoot_721 Backend.backend model params i (w_702_0, (w_703_0, (w_704_0, (w_705_0, (w_706_0, (w_707_0, (w_718_0, (w_718_2, (w_718_4, (w_711_0, (w_718_1, (w_718_3, (w_718_5, (w_700_0, ((w_719_0 i), ((w_720_0 i), ())))))))))))))))) ((w_721_0 i), ((w_721_1 i), ((w_721_2 i), ((w_721_3 i), ((w_721_4 i), ((w_721_5 i), ()))))))))
    (h722_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_722 Backend.backend model params i (w_721_0, (w_721_1, (w_721_2, (w_721_3, (w_721_4, (w_721_5, ())))))) ((w_722_0 i), ((w_722_1 i), ((w_722_2 i), ((w_722_3 i), ((w_722_4 i), ((w_722_5 i), ()))))))))
    (h723_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_723 Backend.backend model params i (w_695_0, (w_696_0, (w_697_0, (w_698_0, (w_699_0, (w_700_0, (w_701_0, ((w_722_4 i), ((w_722_0 i), ((w_722_2 i), ((w_722_5 i), ((w_722_1 i), ((w_722_3 i), ()))))))))))))) ((w_723_0 i), ((w_723_1 i), ((w_723_2 i), ((w_723_3 i), ((w_723_4 i), ((w_723_5 i), ()))))))))
    (h724_0 : familyGetStatic w_723_0 (0) w_724_0)
    (h725_0 : familyGetStatic w_723_0 (1) w_725_0)
    (h727_0 : familyGetStatic w_723_1 (0) w_727_0)
    (h728_0 : familyGetStatic w_723_1 (1) w_728_0)
    (h730_0 : familyGetStatic w_723_2 (0) w_730_0)
    (h731_0 : familyGetStatic w_723_2 (1) w_731_0)
    (h733_0 : familyGetStatic w_723_3 (0) w_733_0)
    (h734_0 : familyGetStatic w_723_3 (1) w_734_0)
    (h736_0 : familyGetStatic w_723_4 (0) w_736_0)
    (h737_0 : familyGetStatic w_723_4 (1) w_737_0)
    (h739_0 : familyGetStatic w_723_5 (0) w_739_0)
    (h740_0 : familyGetStatic w_723_5 (1) w_740_0)
    (h742_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => W_Stage_online_eval_0_1.sequential_generatedRoot_742 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_723_0, (w_723_1, (w_723_2, (w_723_3, (w_723_4, (w_723_5, ())))))))))))) next) (Int.toNat (158)) (w_726_0, (w_729_0, (w_732_0, (w_735_0, (w_738_0, (w_741_0, ())))))) (w_742_0, (w_742_1, (w_742_2, (w_742_3, (w_742_4, (w_742_5, ())))))))
    {sm sp : ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12}
    (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10}
    {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6')
    (hg7 : centeredRebaseRuns g6' g7)
    {g5' : ExactMatrix Q0 N 1 8}
    {h6 : ExactMatrix Q5 N 1 8}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 8 g5')
    (hh6 : centeredRebaseRuns g5' h6)
    {h5 : ExactMatrix Q0 N 1 6}
    {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![w_703_0 k, w_706_0 k], ![w_704_0 k, w_707_0 k]] (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![w_702_0 k, w_705_0 k] Ba)
    (hpay : ∀ k : Fin 5, w_711_0 k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat}
    (hone : Rel sm ![![w_708_0, w_712_0], ![w_709_0, w_713_0]] (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![w_710_0, w_714_0] Bs)
    {s : Fin 800 → Int}
    {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8}
    {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model w_700_0 Q5 8 smudge3Rhs0 i j + ![w_715_0 i, w_716_0 i] j = reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 + reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int}
    {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6}
    {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model w_700_0 Q0 6 smudge3Rhs1 i j + ![w_719_0 i, w_720_0 i] j = reducePoly Q0 N (((q3 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 + reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {g1' : ExactMatrix Q1 N 1 6}
    (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1')
    (hh5 : centeredRebaseRuns g1' h5)
    {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model w_700_0 Q0 6 2 (sigma i)) (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) * unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![w_699_0 i, w_701_0 i] Bh)
    {gq : ExactMatrix Q1 N 1 6}
    (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq)
    {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1)
    {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j, reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) • ![![w_695_0, w_696_0], ![w_697_0, w_698_0]] 0 j i + ![![w_695_0, w_696_0], ![w_697_0, w_698_0]] 1 j i = (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (smudgeTable k)) (sigma i).toNat * reducePoly Q1 N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    Rel sm ![![w_742_0, w_742_1], ![w_742_2, w_742_3]]
      (fun j ↦ reducePoly Q1 N (smudgeValue3 (fun o ↦ a (sel' o) + a (sel o)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![w_742_4, w_742_5] (160 * smudge3LutBound Ba Bs BB BB' Bh BS) := by
  subst e_726_0 e_729_0 e_732_0 e_735_0 e_738_0 e_741_0
  have hs := rootSum742 h724_0 h725_0 h727_0 h728_0 h730_0 h731_0 h733_0 h734_0 h736_0 h737_0 h739_0 h740_0 h742_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  rw [s0, s1, s2, s3, s4, s5]
  exact smudgeTotal3 (model := model) (h717 := h717_5) (h718 := h718_5) (h721 := h721_5) (h722 := h722_5) (hlut := h723_5) (hsm := hsm) (hG := hG) (hg6' := hg6') (hg7 := hg7) (hg5' := hg5') (hh6 := hh6) (ha := ha) (hpay := hpay) (hone := hone) (hrhs := hrhs) (heB := heB) (hrhs' := hrhs') (heB' := heB') (hsel := hsel) (hsel' := hsel') (hg1' := hg1') (hh5 := hh5) (hhelp := hhelp) (hgq := hgq) (hinverse := hinverse) (hswitch := hswitch)


set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The smudge carrier of decoder `4`. -/
theorem smudgeChain4 {model : HashModel} {params : W_Stage_online_eval_0_1.Params}
    {w_749_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_750_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_751_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_752_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_753_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_754_0 : ByteArray}
    {w_755_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_756_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_757_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_758_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_759_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_760_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_761_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_762_0 : ExactMatrix Q6 8 1 10}
    {w_763_0 : ExactMatrix Q6 8 1 10}
    {w_764_0 : ExactMatrix Q6 8 1 10}
    {w_766_0 : ExactMatrix Q6 8 1 10}
    {w_767_0 : ExactMatrix Q6 8 1 10}
    {w_768_0 : ExactMatrix Q6 8 1 10}
    {w_769_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_770_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_773_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_774_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_765_0 : Fin 5 → ExactMatrix Q6 8 1 1}
    {w_771_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_771_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_771_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_771_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_771_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_771_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_772_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_772_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_772_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_772_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_772_4 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_772_5 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_775_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_775_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_775_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_775_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_775_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_775_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_776_0 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_776_1 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_776_2 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_776_3 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_776_4 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_776_5 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_777_0 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_777_1 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_777_2 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_777_3 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_777_4 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_777_5 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_778_0 : ExactMatrix Q1 8 1 1}
    {w_779_0 : ExactMatrix Q1 8 1 1}
    {w_780_0 : ExactMatrix Q1 8 1 1}
    {w_781_0 : ExactMatrix Q1 8 1 1}
    {w_782_0 : ExactMatrix Q1 8 1 1}
    {w_783_0 : ExactMatrix Q1 8 1 1}
    {w_784_0 : ExactMatrix Q1 8 1 1}
    {w_785_0 : ExactMatrix Q1 8 1 1}
    {w_786_0 : ExactMatrix Q1 8 1 1}
    {w_787_0 : ExactMatrix Q1 8 1 1}
    {w_788_0 : ExactMatrix Q1 8 1 1}
    {w_789_0 : ExactMatrix Q1 8 1 1}
    {w_790_0 : ExactMatrix Q1 8 1 1}
    {w_791_0 : ExactMatrix Q1 8 1 1}
    {w_792_0 : ExactMatrix Q1 8 1 1}
    {w_793_0 : ExactMatrix Q1 8 1 1}
    {w_794_0 : ExactMatrix Q1 8 1 1}
    {w_795_0 : ExactMatrix Q1 8 1 1}
    {w_796_0 : ExactMatrix Q1 8 1 1}
    {w_796_1 : ExactMatrix Q1 8 1 1}
    {w_796_2 : ExactMatrix Q1 8 1 1}
    {w_796_3 : ExactMatrix Q1 8 1 1}
    {w_796_4 : ExactMatrix Q1 8 1 1}
    {w_796_5 : ExactMatrix Q1 8 1 1}
    (e_780_0 : w_780_0 = matrixAdd w_778_0 w_779_0)
    (e_783_0 : w_783_0 = matrixAdd w_781_0 w_782_0)
    (e_786_0 : w_786_0 = matrixAdd w_784_0 w_785_0)
    (e_789_0 : w_789_0 = matrixAdd w_787_0 w_788_0)
    (e_792_0 : w_792_0 = matrixAdd w_790_0 w_791_0)
    (e_795_0 : w_795_0 = matrixAdd w_793_0 w_794_0)
    (h771_5 : (∀ i : Fin 800, W_Stage_online_eval_0_1.parallel_generatedRoot_771 Backend.backend model params i (w_756_0, (w_757_0, (w_758_0, (w_759_0, (w_760_0, (w_761_0, (w_762_0, (w_763_0, (w_764_0, (w_765_0, (w_766_0, (w_767_0, (w_768_0, (w_754_0, ((w_769_0 i), ((w_770_0 i), ())))))))))))))))) ((w_771_0 i), ((w_771_1 i), ((w_771_2 i), ((w_771_3 i), ((w_771_4 i), ((w_771_5 i), ()))))))))
    (h772_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_772 Backend.backend model params i (w_771_0, (w_771_1, (w_771_2, (w_771_3, (w_771_4, (w_771_5, ())))))) ((w_772_0 i), ((w_772_1 i), ((w_772_2 i), ((w_772_3 i), ((w_772_4 i), ((w_772_5 i), ()))))))))
    (h775_5 : (∀ i : Fin 800, W_Stage_online_eval_0_1.parallel_generatedRoot_775 Backend.backend model params i (w_756_0, (w_757_0, (w_758_0, (w_759_0, (w_760_0, (w_761_0, (w_772_0, (w_772_2, (w_772_4, (w_765_0, (w_772_1, (w_772_3, (w_772_5, (w_754_0, ((w_773_0 i), ((w_774_0 i), ())))))))))))))))) ((w_775_0 i), ((w_775_1 i), ((w_775_2 i), ((w_775_3 i), ((w_775_4 i), ((w_775_5 i), ()))))))))
    (h776_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_776 Backend.backend model params i (w_775_0, (w_775_1, (w_775_2, (w_775_3, (w_775_4, (w_775_5, ())))))) ((w_776_0 i), ((w_776_1 i), ((w_776_2 i), ((w_776_3 i), ((w_776_4 i), ((w_776_5 i), ()))))))))
    (h777_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_777 Backend.backend model params i (w_749_0, (w_750_0, (w_751_0, (w_752_0, (w_753_0, (w_754_0, (w_755_0, ((w_776_4 i), ((w_776_0 i), ((w_776_2 i), ((w_776_5 i), ((w_776_1 i), ((w_776_3 i), ()))))))))))))) ((w_777_0 i), ((w_777_1 i), ((w_777_2 i), ((w_777_3 i), ((w_777_4 i), ((w_777_5 i), ()))))))))
    (h778_0 : familyGetStatic w_777_0 (0) w_778_0)
    (h779_0 : familyGetStatic w_777_0 (1) w_779_0)
    (h781_0 : familyGetStatic w_777_1 (0) w_781_0)
    (h782_0 : familyGetStatic w_777_1 (1) w_782_0)
    (h784_0 : familyGetStatic w_777_2 (0) w_784_0)
    (h785_0 : familyGetStatic w_777_2 (1) w_785_0)
    (h787_0 : familyGetStatic w_777_3 (0) w_787_0)
    (h788_0 : familyGetStatic w_777_3 (1) w_788_0)
    (h790_0 : familyGetStatic w_777_4 (0) w_790_0)
    (h791_0 : familyGetStatic w_777_4 (1) w_791_0)
    (h793_0 : familyGetStatic w_777_5 (0) w_793_0)
    (h794_0 : familyGetStatic w_777_5 (1) w_794_0)
    (h796_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => W_Stage_online_eval_0_1.sequential_generatedRoot_796 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_777_0, (w_777_1, (w_777_2, (w_777_3, (w_777_4, (w_777_5, ())))))))))))) next) (Int.toNat (158)) (w_780_0, (w_783_0, (w_786_0, (w_789_0, (w_792_0, (w_795_0, ())))))) (w_796_0, (w_796_1, (w_796_2, (w_796_3, (w_796_4, (w_796_5, ())))))))
    {sm sp : ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12}
    (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10}
    {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6')
    (hg7 : centeredRebaseRuns g6' g7)
    {g5' : ExactMatrix Q0 N 1 8}
    {h6 : ExactMatrix Q5 N 1 8}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 8 g5')
    (hh6 : centeredRebaseRuns g5' h6)
    {h5 : ExactMatrix Q0 N 1 6}
    {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![w_757_0 k, w_760_0 k], ![w_758_0 k, w_761_0 k]] (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![w_756_0 k, w_759_0 k] Ba)
    (hpay : ∀ k : Fin 5, w_765_0 k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat}
    (hone : Rel sm ![![w_762_0, w_766_0], ![w_763_0, w_767_0]] (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![w_764_0, w_768_0] Bs)
    {s : Fin 800 → Int}
    {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8}
    {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model w_754_0 Q5 8 smudge4Rhs0 i j + ![w_769_0 i, w_770_0 i] j = reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 + reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int}
    {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6}
    {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model w_754_0 Q0 6 smudge4Rhs1 i j + ![w_773_0 i, w_774_0 i] j = reducePoly Q0 N (((q3 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 + reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {g1' : ExactMatrix Q1 N 1 6}
    (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1')
    (hh5 : centeredRebaseRuns g1' h5)
    {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model w_754_0 Q0 6 2 (sigma i)) (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) * unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![w_753_0 i, w_755_0 i] Bh)
    {gq : ExactMatrix Q1 N 1 6}
    (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq)
    {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1)
    {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j, reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) • ![![w_749_0, w_750_0], ![w_751_0, w_752_0]] 0 j i + ![![w_749_0, w_750_0], ![w_751_0, w_752_0]] 1 j i = (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (smudgeTable k)) (sigma i).toNat * reducePoly Q1 N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    Rel sm ![![w_796_0, w_796_1], ![w_796_2, w_796_3]]
      (fun j ↦ reducePoly Q1 N (smudgeValue4 (fun o ↦ a (sel' o) + a (sel o)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![w_796_4, w_796_5] (160 * smudge4LutBound Ba Bs BB BB' Bh BS) := by
  subst e_780_0 e_783_0 e_786_0 e_789_0 e_792_0 e_795_0
  have hs := rootSum796 h778_0 h779_0 h781_0 h782_0 h784_0 h785_0 h787_0 h788_0 h790_0 h791_0 h793_0 h794_0 h796_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  rw [s0, s1, s2, s3, s4, s5]
  exact smudgeTotal4 (model := model) (h771 := h771_5) (h772 := h772_5) (h775 := h775_5) (h776 := h776_5) (hlut := h777_5) (hsm := hsm) (hG := hG) (hg6' := hg6') (hg7 := hg7) (hg5' := hg5') (hh6 := hh6) (ha := ha) (hpay := hpay) (hone := hone) (hrhs := hrhs) (heB := heB) (hrhs' := hrhs') (heB' := heB') (hsel := hsel) (hsel' := hsel') (hg1' := hg1') (hh5 := hh5) (hhelp := hhelp) (hgq := hgq) (hinverse := hinverse) (hswitch := hswitch)


set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The smudge carrier of decoder `5`. -/
theorem smudgeChain5 {model : HashModel} {params : W_Stage_online_eval_0_1.Params}
    {w_803_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_804_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_805_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_806_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_807_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_808_0 : ByteArray}
    {w_809_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_810_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_811_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_812_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_813_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_814_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_815_0 : Fin 5 → ExactMatrix Q6 8 1 12}
    {w_816_0 : ExactMatrix Q6 8 1 10}
    {w_817_0 : ExactMatrix Q6 8 1 10}
    {w_818_0 : ExactMatrix Q6 8 1 10}
    {w_820_0 : ExactMatrix Q6 8 1 10}
    {w_821_0 : ExactMatrix Q6 8 1 10}
    {w_822_0 : ExactMatrix Q6 8 1 10}
    {w_823_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_824_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_827_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_828_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_819_0 : Fin 5 → ExactMatrix Q6 8 1 1}
    {w_825_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_825_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_825_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_825_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_825_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_825_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {w_826_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_826_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_826_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_826_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_826_4 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_826_5 : Fin 160 → ExactMatrix Q5 8 1 8}
    {w_829_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_829_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_829_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_829_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_829_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_829_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {w_830_0 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_830_1 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_830_2 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_830_3 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_830_4 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_830_5 : Fin 160 → ExactMatrix Q0 8 1 6}
    {w_831_0 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_831_1 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_831_2 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_831_3 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_831_4 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_831_5 : Fin 160 → ExactMatrix Q1 8 1 1}
    {w_832_0 : ExactMatrix Q1 8 1 1}
    {w_833_0 : ExactMatrix Q1 8 1 1}
    {w_834_0 : ExactMatrix Q1 8 1 1}
    {w_835_0 : ExactMatrix Q1 8 1 1}
    {w_836_0 : ExactMatrix Q1 8 1 1}
    {w_837_0 : ExactMatrix Q1 8 1 1}
    {w_838_0 : ExactMatrix Q1 8 1 1}
    {w_839_0 : ExactMatrix Q1 8 1 1}
    {w_840_0 : ExactMatrix Q1 8 1 1}
    {w_841_0 : ExactMatrix Q1 8 1 1}
    {w_842_0 : ExactMatrix Q1 8 1 1}
    {w_843_0 : ExactMatrix Q1 8 1 1}
    {w_844_0 : ExactMatrix Q1 8 1 1}
    {w_845_0 : ExactMatrix Q1 8 1 1}
    {w_846_0 : ExactMatrix Q1 8 1 1}
    {w_847_0 : ExactMatrix Q1 8 1 1}
    {w_848_0 : ExactMatrix Q1 8 1 1}
    {w_849_0 : ExactMatrix Q1 8 1 1}
    {w_850_0 : ExactMatrix Q1 8 1 1}
    {w_850_1 : ExactMatrix Q1 8 1 1}
    {w_850_2 : ExactMatrix Q1 8 1 1}
    {w_850_3 : ExactMatrix Q1 8 1 1}
    {w_850_4 : ExactMatrix Q1 8 1 1}
    {w_850_5 : ExactMatrix Q1 8 1 1}
    (e_834_0 : w_834_0 = matrixAdd w_832_0 w_833_0)
    (e_837_0 : w_837_0 = matrixAdd w_835_0 w_836_0)
    (e_840_0 : w_840_0 = matrixAdd w_838_0 w_839_0)
    (e_843_0 : w_843_0 = matrixAdd w_841_0 w_842_0)
    (e_846_0 : w_846_0 = matrixAdd w_844_0 w_845_0)
    (e_849_0 : w_849_0 = matrixAdd w_847_0 w_848_0)
    (h825_5 : (∀ i : Fin 800, W_Stage_online_eval_0_1.parallel_generatedRoot_825 Backend.backend model params i (w_810_0, (w_811_0, (w_812_0, (w_813_0, (w_814_0, (w_815_0, (w_816_0, (w_817_0, (w_818_0, (w_819_0, (w_820_0, (w_821_0, (w_822_0, (w_808_0, ((w_823_0 i), ((w_824_0 i), ())))))))))))))))) ((w_825_0 i), ((w_825_1 i), ((w_825_2 i), ((w_825_3 i), ((w_825_4 i), ((w_825_5 i), ()))))))))
    (h826_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_826 Backend.backend model params i (w_825_0, (w_825_1, (w_825_2, (w_825_3, (w_825_4, (w_825_5, ())))))) ((w_826_0 i), ((w_826_1 i), ((w_826_2 i), ((w_826_3 i), ((w_826_4 i), ((w_826_5 i), ()))))))))
    (h829_5 : (∀ i : Fin 800, W_Stage_online_eval_0_1.parallel_generatedRoot_829 Backend.backend model params i (w_810_0, (w_811_0, (w_812_0, (w_813_0, (w_814_0, (w_815_0, (w_826_0, (w_826_2, (w_826_4, (w_819_0, (w_826_1, (w_826_3, (w_826_5, (w_808_0, ((w_827_0 i), ((w_828_0 i), ())))))))))))))))) ((w_829_0 i), ((w_829_1 i), ((w_829_2 i), ((w_829_3 i), ((w_829_4 i), ((w_829_5 i), ()))))))))
    (h830_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_830 Backend.backend model params i (w_829_0, (w_829_1, (w_829_2, (w_829_3, (w_829_4, (w_829_5, ())))))) ((w_830_0 i), ((w_830_1 i), ((w_830_2 i), ((w_830_3 i), ((w_830_4 i), ((w_830_5 i), ()))))))))
    (h831_5 : (∀ i : Fin 160, W_Stage_online_eval_0_1.parallel_generatedRoot_831 Backend.backend model params i (w_803_0, (w_804_0, (w_805_0, (w_806_0, (w_807_0, (w_808_0, (w_809_0, ((w_830_4 i), ((w_830_0 i), ((w_830_2 i), ((w_830_5 i), ((w_830_1 i), ((w_830_3 i), ()))))))))))))) ((w_831_0 i), ((w_831_1 i), ((w_831_2 i), ((w_831_3 i), ((w_831_4 i), ((w_831_5 i), ()))))))))
    (h832_0 : familyGetStatic w_831_0 (0) w_832_0)
    (h833_0 : familyGetStatic w_831_0 (1) w_833_0)
    (h835_0 : familyGetStatic w_831_1 (0) w_835_0)
    (h836_0 : familyGetStatic w_831_1 (1) w_836_0)
    (h838_0 : familyGetStatic w_831_2 (0) w_838_0)
    (h839_0 : familyGetStatic w_831_2 (1) w_839_0)
    (h841_0 : familyGetStatic w_831_3 (0) w_841_0)
    (h842_0 : familyGetStatic w_831_3 (1) w_842_0)
    (h844_0 : familyGetStatic w_831_4 (0) w_844_0)
    (h845_0 : familyGetStatic w_831_4 (1) w_845_0)
    (h847_0 : familyGetStatic w_831_5 (0) w_847_0)
    (h848_0 : familyGetStatic w_831_5 (1) w_848_0)
    (h850_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => W_Stage_online_eval_0_1.sequential_generatedRoot_850 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_831_0, (w_831_1, (w_831_2, (w_831_3, (w_831_4, (w_831_5, ())))))))))))) next) (Int.toNat (158)) (w_834_0, (w_837_0, (w_840_0, (w_843_0, (w_846_0, (w_849_0, ())))))) (w_850_0, (w_850_1, (w_850_2, (w_850_3, (w_850_4, (w_850_5, ())))))))
    {sm sp : ErrorPoly N}
    (hsm : polyNorm sm ≤ 1)
    {a : Fin 5 → Nat}
    {G : ExactMatrix Q6 N 1 12}
    (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    {g6' : ExactMatrix Q5 N 1 10}
    {g7 : ExactMatrix Q6 N 1 10}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 10 g6')
    (hg7 : centeredRebaseRuns g6' g7)
    {g5' : ExactMatrix Q0 N 1 8}
    {h6 : ExactMatrix Q5 N 1 8}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 8 g5')
    (hh6 : centeredRebaseRuns g5' h6)
    {h5 : ExactMatrix Q0 N 1 6}
    {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![w_811_0 k, w_814_0 k], ![w_812_0 k, w_815_0 k]] (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![w_810_0 k, w_813_0 k] Ba)
    (hpay : ∀ k : Fin 5, w_819_0 k = fun _ _ ↦ reducePoly Q6 N (Xe (n := N) (2 * a k)))
    {Bs : Nat}
    (hone : Rel sm ![![w_816_0, w_820_0], ![w_817_0, w_821_0]] (fun j ↦ reducePoly Q6 N (((p5 : Nat) : Int) * 1 * unitPair sp j) • g7) ![w_818_0, w_822_0] Bs)
    {s : Fin 800 → Int}
    {eB : Fin 800 → Fin 2 → ErrorMatrix N 1 8}
    {BB : Nat}
    (hrhs : ∀ (i : Fin 800) j, reducePoly Q5 N sp • rhsTop model w_808_0 Q5 8 smudge5Rhs0 i j + ![w_823_0 i, w_824_0 i] j = reducePoly Q5 N (((p4 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 + reduceMatrix Q5 N 1 8 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 800 → Int}
    {eB' : Fin 800 → Fin 2 → ErrorMatrix N 1 6}
    {BB' : Nat}
    (hrhs' : ∀ (i : Fin 800) j, reducePoly Q0 N sp • rhsTop model w_808_0 Q0 6 smudge5Rhs1 i j + ![w_827_0 i, w_828_0 i] j = reducePoly Q0 N (((q3 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 + reduceMatrix Q0 N 1 6 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 160 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {g1' : ExactMatrix Q1 N 1 6}
    (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1')
    (hh5 : centeredRebaseRuns g1' h5)
    {Bh : Nat}
    (hhelp : ∀ i : Fin 4, Rel sm (lutA model w_808_0 Q0 6 2 (sigma i)) (fun j ↦ reducePoly Q0 N (((q3 : Nat) : Int) * unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![w_807_0 i, w_809_0 i] Bh)
    {gq : ExactMatrix Q1 N 1 6}
    (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq)
    {inverse : Nat}
    (hinverse : (inverse : ExactPoly Q1 N) * ((4 : Nat) : ExactPoly Q1 N) = 1)
    {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j, reducePoly Q1 N (substitution (sigma i).toNat (sigma_odd i) sp) • ![![w_803_0, w_804_0], ![w_805_0, w_806_0]] 0 j i + ![![w_803_0, w_804_0], ![w_805_0, w_806_0]] 1 j i = (lutCoefficient Q1 inverse (fun k ↦ reducePoly Q1 N (smudgeTable k)) (sigma i).toNat * reducePoly Q1 N ((1 : ErrorPoly N) * unitPair sp j)) • gq + reduceMatrix Q1 N 1 6 (e j)) :
    Rel sm ![![w_850_0, w_850_1], ![w_850_2, w_850_3]]
      (fun j ↦ reducePoly Q1 N (smudgeValue5 (fun o ↦ a (sel' o) + a (sel o)) * unitPair sp j) •
        (1 : ExactMatrix Q1 N 1 1))
      ![w_850_4, w_850_5] (160 * smudge5LutBound Ba Bs BB BB' Bh BS) := by
  subst e_834_0 e_837_0 e_840_0 e_843_0 e_846_0 e_849_0
  have hs := rootSum850 h832_0 h833_0 h835_0 h836_0 h838_0 h839_0 h841_0 h842_0 h844_0 h845_0 h847_0 h848_0 h850_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  rw [s0, s1, s2, s3, s4, s5]
  exact smudgeTotal5 (model := model) (h825 := h825_5) (h826 := h826_5) (h829 := h829_5) (h830 := h830_5) (hlut := h831_5) (hsm := hsm) (hG := hG) (hg6' := hg6') (hg7 := hg7) (hg5' := hg5') (hh6 := hh6) (ha := ha) (hpay := hpay) (hone := hone) (hrhs := hrhs) (heB := heB) (hrhs' := hrhs') (heB' := heB') (hsel := hsel) (hsel' := hsel') (hg1' := hg1') (hh5 := hh5) (hhelp := hhelp) (hgq := hgq) (hinverse := hinverse) (hswitch := hswitch)


end RefreshTopProof.Part_0_1
