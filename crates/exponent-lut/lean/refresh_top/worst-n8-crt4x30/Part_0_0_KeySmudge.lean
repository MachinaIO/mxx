import Part_0_0_KeysLuts
import Part_0_0_KeysCells
import Part_0_0_Sums
import Part_0_0_PreSums
import Part_0_0_PreRootSums
import Part_0_0_RootSums

/-! The smudge keys of the online root of part `0_0` are those the preprocessing computes
(written by `generator/gen_key_links.py`). -/

namespace RefreshTopProof.Part_0_0

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The smudge keys of decoder `0`: the online root's are the preprocessing's. -/
theorem smudgeKeyLink0 {model : HashModel} {params : Stage_online_eval_0_0.Params} {params' : Stage_preprocess_0_0.Params}
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
    (h554_5 : (∀ i : Fin 800, Stage_online_eval_0_0.parallel_generatedRoot_554 Backend.backend model params i (w_539_0, (w_540_0, (w_541_0, (w_542_0, (w_543_0, (w_544_0, (w_545_0, (w_546_0, (w_547_0, (w_548_0, (w_549_0, (w_550_0, (w_551_0, (w_537_0, ((w_552_0 i), ((w_553_0 i), ())))))))))))))))) ((w_554_0 i), ((w_554_1 i), ((w_554_2 i), ((w_554_3 i), ((w_554_4 i), ((w_554_5 i), ()))))))))
    (h555_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_555 Backend.backend model params i (w_554_0, (w_554_1, (w_554_2, (w_554_3, (w_554_4, (w_554_5, ())))))) ((w_555_0 i), ((w_555_1 i), ((w_555_2 i), ((w_555_3 i), ((w_555_4 i), ((w_555_5 i), ()))))))))
    (h558_5 : (∀ i : Fin 800, Stage_online_eval_0_0.parallel_generatedRoot_558 Backend.backend model params i (w_539_0, (w_540_0, (w_541_0, (w_542_0, (w_543_0, (w_544_0, (w_555_0, (w_555_2, (w_555_4, (w_548_0, (w_555_1, (w_555_3, (w_555_5, (w_537_0, ((w_556_0 i), ((w_557_0 i), ())))))))))))))))) ((w_558_0 i), ((w_558_1 i), ((w_558_2 i), ((w_558_3 i), ((w_558_4 i), ((w_558_5 i), ()))))))))
    (h559_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_559 Backend.backend model params i (w_558_0, (w_558_1, (w_558_2, (w_558_3, (w_558_4, (w_558_5, ())))))) ((w_559_0 i), ((w_559_1 i), ((w_559_2 i), ((w_559_3 i), ((w_559_4 i), ((w_559_5 i), ()))))))))
    (h560_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_560 Backend.backend model params i (w_532_0, (w_533_0, (w_534_0, (w_535_0, (w_536_0, (w_537_0, (w_538_0, ((w_559_4 i), ((w_559_0 i), ((w_559_2 i), ((w_559_5 i), ((w_559_1 i), ((w_559_3 i), ()))))))))))))) ((w_560_0 i), ((w_560_1 i), ((w_560_2 i), ((w_560_3 i), ((w_560_4 i), ((w_560_5 i), ()))))))))
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
    (h579_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => Stage_online_eval_0_0.sequential_generatedRoot_579 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_560_0, (w_560_1, (w_560_2, (w_560_3, (w_560_4, (w_560_5, ())))))))))))) next) (Int.toNat (158)) (w_563_0, (w_566_0, (w_569_0, (w_572_0, (w_575_0, (w_578_0, ())))))) (w_579_0, (w_579_1, (w_579_2, (w_579_3, (w_579_4, (w_579_5, ())))))))
    {pw_363_0 : ByteArray}
    {pw_379_0 : Fin 800 → Int}
    {pw_380_0 : Fin 800 → Int}
    {pw_383_0 : Fin 800 → Int}
    {pw_384_0 : Fin 800 → Int}
    {pw_372_0 : ExactMatrix Q6 8 1 1}
    {pw_373_0 : ExactMatrix Q6 8 1 1}
    {pw_374_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_374_1 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_374_2 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_374_3 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_375_0 : ExactMatrix Q6 8 1 10}
    {pw_376_0 : ExactMatrix Q6 8 1 10}
    {pw_377_0 : ExactMatrix Q6 8 1 10}
    {pw_378_0 : ExactMatrix Q6 8 1 10}
    {pw_381_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_381_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_381_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_381_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_381_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_381_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_381_6 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_381_7 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_382_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_382_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_382_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_382_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_385_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_385_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_385_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_385_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_385_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_385_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_385_6 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_385_7 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_387_0 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_387_1 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_387_2 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_387_3 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_388_0 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_388_1 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_388_2 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_388_3 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_389_0 : ExactMatrix Q1 8 1 1}
    {pw_390_0 : ExactMatrix Q1 8 1 1}
    {pw_391_0 : ExactMatrix Q1 8 1 1}
    {pw_392_0 : ExactMatrix Q1 8 1 1}
    {pw_393_0 : ExactMatrix Q1 8 1 1}
    {pw_394_0 : ExactMatrix Q1 8 1 1}
    {pw_395_0 : ExactMatrix Q1 8 1 1}
    {pw_396_0 : ExactMatrix Q1 8 1 1}
    {pw_397_0 : ExactMatrix Q1 8 1 1}
    {pw_398_0 : ExactMatrix Q1 8 1 1}
    {pw_399_0 : ExactMatrix Q1 8 1 1}
    {pw_400_0 : ExactMatrix Q1 8 1 1}
    {pw_401_0 : ExactMatrix Q1 8 1 1}
    {pw_401_1 : ExactMatrix Q1 8 1 1}
    {pw_401_2 : ExactMatrix Q1 8 1 1}
    {pw_401_3 : ExactMatrix Q1 8 1 1}
    (pe_391_0 : pw_391_0 = matrixAdd pw_389_0 pw_390_0)
    (pe_394_0 : pw_394_0 = matrixAdd pw_392_0 pw_393_0)
    (pe_397_0 : pw_397_0 = matrixAdd pw_395_0 pw_396_0)
    (pe_400_0 : pw_400_0 = matrixAdd pw_398_0 pw_399_0)
    (ph381_7 : (∀ i : Fin 800, Stage_preprocess_0_0.parallel_generatedRoot_381 Backend.backend model params' i (pw_363_0, (pw_375_0, (pw_376_0, (pw_377_0, (pw_378_0, (pw_372_0, ((pw_379_0 i), ((pw_380_0 i), (pw_373_0, ()))))))))) ((pw_381_0 i), ((pw_381_1 i), ((pw_381_2 i), ((pw_381_3 i), ((pw_381_4 i), ((pw_381_5 i), ((pw_381_6 i), ((pw_381_7 i), ()))))))))))
    (ph382_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_382 Backend.backend model params' i (pw_381_0, (pw_381_1, (pw_381_2, (pw_381_3, ())))) ((pw_382_0 i), ((pw_382_1 i), ((pw_382_2 i), ((pw_382_3 i), ()))))))
    (ph385_7 : (∀ i : Fin 800, Stage_preprocess_0_0.parallel_generatedRoot_385 Backend.backend model params' i (pw_363_0, (pw_382_0, (pw_382_2, (pw_382_1, (pw_382_3, (pw_372_0, ((pw_383_0 i), ((pw_384_0 i), (pw_373_0, ()))))))))) ((pw_385_0 i), ((pw_385_1 i), ((pw_385_2 i), ((pw_385_3 i), ((pw_385_4 i), ((pw_385_5 i), ((pw_385_6 i), ((pw_385_7 i), ()))))))))))
    (ph387_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_387 Backend.backend model params' i (pw_385_0, (pw_385_1, (pw_385_2, (pw_385_3, ())))) ((pw_387_0 i), ((pw_387_1 i), ((pw_387_2 i), ((pw_387_3 i), ()))))))
    (ph388_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_388 Backend.backend model params' i (pw_363_0, ((pw_387_0 i), ((pw_387_2 i), (pw_374_0, ((pw_387_1 i), ((pw_387_3 i), (pw_374_2, (pw_374_1, (pw_374_3, ()))))))))) ((pw_388_0 i), ((pw_388_1 i), ((pw_388_2 i), ((pw_388_3 i), ()))))))
    (ph389_0 : familyGetStatic pw_388_0 (0) pw_389_0)
    (ph390_0 : familyGetStatic pw_388_0 (1) pw_390_0)
    (ph392_0 : familyGetStatic pw_388_1 (0) pw_392_0)
    (ph393_0 : familyGetStatic pw_388_1 (1) pw_393_0)
    (ph395_0 : familyGetStatic pw_388_2 (0) pw_395_0)
    (ph396_0 : familyGetStatic pw_388_2 (1) pw_396_0)
    (ph398_0 : familyGetStatic pw_388_3 (0) pw_398_0)
    (ph399_0 : familyGetStatic pw_388_3 (1) pw_399_0)
    (ph401_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => Stage_preprocess_0_0.sequential_generatedRoot_401 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_388_0, (pw_388_1, (pw_388_2, (pw_388_3, ())))))))) next) (Int.toNat (158)) (pw_391_0, (pw_394_0, (pw_397_0, (pw_400_0, ())))) (pw_401_0, (pw_401_1, (pw_401_2, (pw_401_3, ())))))
    (hk : w_537_0 = pw_363_0)
    (ho0 : w_545_0 = pw_375_0) (ho1 : w_546_0 = pw_376_0) (ho2 : w_549_0 = pw_377_0) (ho3 : w_550_0 = pw_378_0)
    (hf0 : w_532_0 = pw_374_0) (hf1 : w_533_0 = pw_374_1) (hf2 : w_534_0 = pw_374_2) (hf3 : w_535_0 = pw_374_3)
    (hc0 : w_552_0 = pw_381_6) (hc1 : w_553_0 = pw_381_7)
    (hc2 : w_556_0 = pw_385_6) (hc3 : w_557_0 = pw_385_7)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![w_540_0, w_543_0, w_541_0, w_544_0] c k =
      model.sample Q6 N 1 12 w_537_0 (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)])) :
    w_579_1 = pw_401_1 ∧ w_579_3 = pw_401_3 := by
  subst hk ho0 ho1 ho2 ho3 hf0 hf1 hf2 hf3 hc0 hc1 hc2 hc3
  have hc554 : ∀ i : Fin 800, w_554_0 i = pw_381_0 i ∧ w_554_1 i = pw_381_1 i ∧
      w_554_2 i = pw_381_2 i ∧ w_554_3 i = pw_381_3 i := fun i ↦
    keysCell554 (hon := h554_5 i) (hpre := ph381_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm554 : w_555_0 = pw_382_0 ∧ w_555_1 = pw_382_1 ∧ w_555_2 = pw_382_2 ∧ w_555_3 = pw_382_3 := by
    have a := fun o ↦ members555 (h555_5 o)
    have b := fun o ↦ premembers382 (ph382_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc554 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc554 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc554 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc554 _).2.2.2
  obtain ⟨m0, m1, m2, m3⟩ := hm554
  rw [m0, m1, m2, m3] at h558_5
  have hc558 : ∀ i : Fin 800, w_558_0 i = pw_385_0 i ∧ w_558_1 i = pw_385_1 i ∧
      w_558_2 i = pw_385_2 i ∧ w_558_3 i = pw_385_3 i := fun i ↦
    keysCell558 (hon := h558_5 i) (hpre := ph385_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm558 : w_559_0 = pw_387_0 ∧ w_559_1 = pw_387_1 ∧ w_559_2 = pw_387_2 ∧ w_559_3 = pw_387_3 := by
    have a := fun o ↦ members559 (h559_5 o)
    have b := fun o ↦ premembers387 (ph387_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc558 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc558 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc558 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc558 _).2.2.2
  obtain ⟨n0, n1, n2, n3⟩ := hm558
  rw [n0, n1, n2, n3] at h560_5
  have hl : ∀ o : Fin 160, w_560_0 o = pw_388_0 o ∧ w_560_1 o = pw_388_1 o ∧ w_560_2 o = pw_388_2 o ∧
      w_560_3 o = pw_388_3 o := by
    intro o
    have a := keyRound560 (h560_5 o)
    have b := preKeyRound388 (ph388_3 o)
    rw [← b] at a
    simp only [Prod.mk.injEq] at a
    exact ⟨a.1, a.2.1, a.2.2.1, a.2.2.2.1⟩
  subst e_563_0 e_566_0 e_569_0 e_572_0 e_575_0 e_578_0 pe_391_0 pe_394_0 pe_397_0 pe_400_0
  have a := rootSum579 h561_0 h562_0 h564_0 h565_0 h567_0 h568_0 h570_0 h571_0 h573_0 h574_0 h576_0 h577_0 h579_5
  have b := prerootSum401 ph389_0 ph390_0 ph392_0 ph393_0 ph395_0 ph396_0 ph398_0 ph399_0 ph401_3
  simp only [Prod.mk.injEq] at a b
  rw [a.2.1, a.2.2.2.1, b.2.1, b.2.2.2.1]
  exact ⟨Finset.sum_congr rfl fun o _ ↦ (hl o).2.1, Finset.sum_congr rfl fun o _ ↦ (hl o).2.2.2⟩

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The smudge keys of decoder `1`: the online root's are the preprocessing's. -/
theorem smudgeKeyLink1 {model : HashModel} {params : Stage_online_eval_0_0.Params} {params' : Stage_preprocess_0_0.Params}
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
    (h609_5 : (∀ i : Fin 800, Stage_online_eval_0_0.parallel_generatedRoot_609 Backend.backend model params i (w_594_0, (w_595_0, (w_596_0, (w_597_0, (w_598_0, (w_599_0, (w_600_0, (w_601_0, (w_602_0, (w_603_0, (w_604_0, (w_605_0, (w_606_0, (w_592_0, ((w_607_0 i), ((w_608_0 i), ())))))))))))))))) ((w_609_0 i), ((w_609_1 i), ((w_609_2 i), ((w_609_3 i), ((w_609_4 i), ((w_609_5 i), ()))))))))
    (h610_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_610 Backend.backend model params i (w_609_0, (w_609_1, (w_609_2, (w_609_3, (w_609_4, (w_609_5, ())))))) ((w_610_0 i), ((w_610_1 i), ((w_610_2 i), ((w_610_3 i), ((w_610_4 i), ((w_610_5 i), ()))))))))
    (h613_5 : (∀ i : Fin 800, Stage_online_eval_0_0.parallel_generatedRoot_613 Backend.backend model params i (w_594_0, (w_595_0, (w_596_0, (w_597_0, (w_598_0, (w_599_0, (w_610_0, (w_610_2, (w_610_4, (w_603_0, (w_610_1, (w_610_3, (w_610_5, (w_592_0, ((w_611_0 i), ((w_612_0 i), ())))))))))))))))) ((w_613_0 i), ((w_613_1 i), ((w_613_2 i), ((w_613_3 i), ((w_613_4 i), ((w_613_5 i), ()))))))))
    (h614_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_614 Backend.backend model params i (w_613_0, (w_613_1, (w_613_2, (w_613_3, (w_613_4, (w_613_5, ())))))) ((w_614_0 i), ((w_614_1 i), ((w_614_2 i), ((w_614_3 i), ((w_614_4 i), ((w_614_5 i), ()))))))))
    (h615_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_615 Backend.backend model params i (w_587_0, (w_588_0, (w_589_0, (w_590_0, (w_591_0, (w_592_0, (w_593_0, ((w_614_4 i), ((w_614_0 i), ((w_614_2 i), ((w_614_5 i), ((w_614_1 i), ((w_614_3 i), ()))))))))))))) ((w_615_0 i), ((w_615_1 i), ((w_615_2 i), ((w_615_3 i), ((w_615_4 i), ((w_615_5 i), ()))))))))
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
    (h634_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => Stage_online_eval_0_0.sequential_generatedRoot_634 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_615_0, (w_615_1, (w_615_2, (w_615_3, (w_615_4, (w_615_5, ())))))))))))) next) (Int.toNat (158)) (w_618_0, (w_621_0, (w_624_0, (w_627_0, (w_630_0, (w_633_0, ())))))) (w_634_0, (w_634_1, (w_634_2, (w_634_3, (w_634_4, (w_634_5, ())))))))
    {pw_0_0 : ByteArray}
    {pw_18_0 : Fin 800 → Int}
    {pw_19_0 : Fin 800 → Int}
    {pw_22_0 : Fin 800 → Int}
    {pw_23_0 : Fin 800 → Int}
    {pw_10_0 : ExactMatrix Q6 8 1 1}
    {pw_12_0 : ExactMatrix Q6 8 1 1}
    {pw_13_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_13_1 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_13_2 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_13_3 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_14_0 : ExactMatrix Q6 8 1 10}
    {pw_15_0 : ExactMatrix Q6 8 1 10}
    {pw_16_0 : ExactMatrix Q6 8 1 10}
    {pw_17_0 : ExactMatrix Q6 8 1 10}
    {pw_20_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_20_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_20_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_20_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_20_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_20_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_20_6 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_20_7 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_21_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_21_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_21_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_21_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_24_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_24_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_24_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_24_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_24_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_24_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_24_6 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_24_7 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_322_0 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_322_1 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_322_2 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_322_3 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_323_0 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_323_1 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_323_2 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_323_3 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_324_0 : ExactMatrix Q1 8 1 1}
    {pw_325_0 : ExactMatrix Q1 8 1 1}
    {pw_326_0 : ExactMatrix Q1 8 1 1}
    {pw_327_0 : ExactMatrix Q1 8 1 1}
    {pw_328_0 : ExactMatrix Q1 8 1 1}
    {pw_329_0 : ExactMatrix Q1 8 1 1}
    {pw_330_0 : ExactMatrix Q1 8 1 1}
    {pw_331_0 : ExactMatrix Q1 8 1 1}
    {pw_332_0 : ExactMatrix Q1 8 1 1}
    {pw_333_0 : ExactMatrix Q1 8 1 1}
    {pw_334_0 : ExactMatrix Q1 8 1 1}
    {pw_335_0 : ExactMatrix Q1 8 1 1}
    {pw_336_0 : ExactMatrix Q1 8 1 1}
    {pw_336_1 : ExactMatrix Q1 8 1 1}
    {pw_336_2 : ExactMatrix Q1 8 1 1}
    {pw_336_3 : ExactMatrix Q1 8 1 1}
    (pe_326_0 : pw_326_0 = matrixAdd pw_324_0 pw_325_0)
    (pe_329_0 : pw_329_0 = matrixAdd pw_327_0 pw_328_0)
    (pe_332_0 : pw_332_0 = matrixAdd pw_330_0 pw_331_0)
    (pe_335_0 : pw_335_0 = matrixAdd pw_333_0 pw_334_0)
    (ph20_7 : (∀ i : Fin 800, Stage_preprocess_0_0.parallel_generatedRoot_20 Backend.backend model params' i (pw_0_0, (pw_14_0, (pw_15_0, (pw_16_0, (pw_17_0, (pw_10_0, ((pw_18_0 i), ((pw_19_0 i), (pw_12_0, ()))))))))) ((pw_20_0 i), ((pw_20_1 i), ((pw_20_2 i), ((pw_20_3 i), ((pw_20_4 i), ((pw_20_5 i), ((pw_20_6 i), ((pw_20_7 i), ()))))))))))
    (ph21_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_21 Backend.backend model params' i (pw_20_0, (pw_20_1, (pw_20_2, (pw_20_3, ())))) ((pw_21_0 i), ((pw_21_1 i), ((pw_21_2 i), ((pw_21_3 i), ()))))))
    (ph24_7 : (∀ i : Fin 800, Stage_preprocess_0_0.parallel_generatedRoot_24 Backend.backend model params' i (pw_0_0, (pw_21_0, (pw_21_2, (pw_21_1, (pw_21_3, (pw_10_0, ((pw_22_0 i), ((pw_23_0 i), (pw_12_0, ()))))))))) ((pw_24_0 i), ((pw_24_1 i), ((pw_24_2 i), ((pw_24_3 i), ((pw_24_4 i), ((pw_24_5 i), ((pw_24_6 i), ((pw_24_7 i), ()))))))))))
    (ph322_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_322 Backend.backend model params' i (pw_24_0, (pw_24_1, (pw_24_2, (pw_24_3, ())))) ((pw_322_0 i), ((pw_322_1 i), ((pw_322_2 i), ((pw_322_3 i), ()))))))
    (ph323_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_323 Backend.backend model params' i (pw_0_0, ((pw_322_0 i), ((pw_322_2 i), (pw_13_0, ((pw_322_1 i), ((pw_322_3 i), (pw_13_2, (pw_13_1, (pw_13_3, ()))))))))) ((pw_323_0 i), ((pw_323_1 i), ((pw_323_2 i), ((pw_323_3 i), ()))))))
    (ph324_0 : familyGetStatic pw_323_0 (0) pw_324_0)
    (ph325_0 : familyGetStatic pw_323_0 (1) pw_325_0)
    (ph327_0 : familyGetStatic pw_323_1 (0) pw_327_0)
    (ph328_0 : familyGetStatic pw_323_1 (1) pw_328_0)
    (ph330_0 : familyGetStatic pw_323_2 (0) pw_330_0)
    (ph331_0 : familyGetStatic pw_323_2 (1) pw_331_0)
    (ph333_0 : familyGetStatic pw_323_3 (0) pw_333_0)
    (ph334_0 : familyGetStatic pw_323_3 (1) pw_334_0)
    (ph336_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => Stage_preprocess_0_0.sequential_generatedRoot_336 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_323_0, (pw_323_1, (pw_323_2, (pw_323_3, ())))))))) next) (Int.toNat (158)) (pw_326_0, (pw_329_0, (pw_332_0, (pw_335_0, ())))) (pw_336_0, (pw_336_1, (pw_336_2, (pw_336_3, ())))))
    (hk : w_592_0 = pw_0_0)
    (ho0 : w_600_0 = pw_14_0) (ho1 : w_601_0 = pw_15_0) (ho2 : w_604_0 = pw_16_0) (ho3 : w_605_0 = pw_17_0)
    (hf0 : w_587_0 = pw_13_0) (hf1 : w_588_0 = pw_13_1) (hf2 : w_589_0 = pw_13_2) (hf3 : w_590_0 = pw_13_3)
    (hc0 : w_607_0 = pw_20_6) (hc1 : w_608_0 = pw_20_7)
    (hc2 : w_611_0 = pw_24_6) (hc3 : w_612_0 = pw_24_7)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![w_595_0, w_598_0, w_596_0, w_599_0] c k =
      model.sample Q6 N 1 12 w_592_0 (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)])) :
    w_634_1 = pw_336_1 ∧ w_634_3 = pw_336_3 := by
  subst hk ho0 ho1 ho2 ho3 hf0 hf1 hf2 hf3 hc0 hc1 hc2 hc3
  have hc609 : ∀ i : Fin 800, w_609_0 i = pw_20_0 i ∧ w_609_1 i = pw_20_1 i ∧
      w_609_2 i = pw_20_2 i ∧ w_609_3 i = pw_20_3 i := fun i ↦
    keysCell609 (hon := h609_5 i) (hpre := ph20_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm609 : w_610_0 = pw_21_0 ∧ w_610_1 = pw_21_1 ∧ w_610_2 = pw_21_2 ∧ w_610_3 = pw_21_3 := by
    have a := fun o ↦ members610 (h610_5 o)
    have b := fun o ↦ premembers21 (ph21_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc609 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc609 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc609 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc609 _).2.2.2
  obtain ⟨m0, m1, m2, m3⟩ := hm609
  rw [m0, m1, m2, m3] at h613_5
  have hc613 : ∀ i : Fin 800, w_613_0 i = pw_24_0 i ∧ w_613_1 i = pw_24_1 i ∧
      w_613_2 i = pw_24_2 i ∧ w_613_3 i = pw_24_3 i := fun i ↦
    keysCell613 (hon := h613_5 i) (hpre := ph24_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm613 : w_614_0 = pw_322_0 ∧ w_614_1 = pw_322_1 ∧ w_614_2 = pw_322_2 ∧ w_614_3 = pw_322_3 := by
    have a := fun o ↦ members614 (h614_5 o)
    have b := fun o ↦ premembers322 (ph322_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc613 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc613 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc613 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc613 _).2.2.2
  obtain ⟨n0, n1, n2, n3⟩ := hm613
  rw [n0, n1, n2, n3] at h615_5
  have hl : ∀ o : Fin 160, w_615_0 o = pw_323_0 o ∧ w_615_1 o = pw_323_1 o ∧ w_615_2 o = pw_323_2 o ∧
      w_615_3 o = pw_323_3 o := by
    intro o
    have a := keyRound615 (h615_5 o)
    have b := preKeyRound323 (ph323_3 o)
    rw [show smudgeWeight615 = smudgeWeight560 from rfl] at a
    rw [← b] at a
    simp only [Prod.mk.injEq] at a
    exact ⟨a.1, a.2.1, a.2.2.1, a.2.2.2.1⟩
  subst e_618_0 e_621_0 e_624_0 e_627_0 e_630_0 e_633_0 pe_326_0 pe_329_0 pe_332_0 pe_335_0
  have a := rootSum634 h616_0 h617_0 h619_0 h620_0 h622_0 h623_0 h625_0 h626_0 h628_0 h629_0 h631_0 h632_0 h634_5
  have b := prerootSum336 ph324_0 ph325_0 ph327_0 ph328_0 ph330_0 ph331_0 ph333_0 ph334_0 ph336_3
  simp only [Prod.mk.injEq] at a b
  rw [a.2.1, a.2.2.2.1, b.2.1, b.2.2.2.1]
  exact ⟨Finset.sum_congr rfl fun o _ ↦ (hl o).2.1, Finset.sum_congr rfl fun o _ ↦ (hl o).2.2.2⟩

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The smudge keys of decoder `2`: the online root's are the preprocessing's. -/
theorem smudgeKeyLink2 {model : HashModel} {params : Stage_online_eval_0_0.Params} {params' : Stage_preprocess_0_0.Params}
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
    (h663_5 : (∀ i : Fin 800, Stage_online_eval_0_0.parallel_generatedRoot_663 Backend.backend model params i (w_648_0, (w_649_0, (w_650_0, (w_651_0, (w_652_0, (w_653_0, (w_654_0, (w_655_0, (w_656_0, (w_657_0, (w_658_0, (w_659_0, (w_660_0, (w_646_0, ((w_661_0 i), ((w_662_0 i), ())))))))))))))))) ((w_663_0 i), ((w_663_1 i), ((w_663_2 i), ((w_663_3 i), ((w_663_4 i), ((w_663_5 i), ()))))))))
    (h664_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_664 Backend.backend model params i (w_663_0, (w_663_1, (w_663_2, (w_663_3, (w_663_4, (w_663_5, ())))))) ((w_664_0 i), ((w_664_1 i), ((w_664_2 i), ((w_664_3 i), ((w_664_4 i), ((w_664_5 i), ()))))))))
    (h667_5 : (∀ i : Fin 800, Stage_online_eval_0_0.parallel_generatedRoot_667 Backend.backend model params i (w_648_0, (w_649_0, (w_650_0, (w_651_0, (w_652_0, (w_653_0, (w_664_0, (w_664_2, (w_664_4, (w_657_0, (w_664_1, (w_664_3, (w_664_5, (w_646_0, ((w_665_0 i), ((w_666_0 i), ())))))))))))))))) ((w_667_0 i), ((w_667_1 i), ((w_667_2 i), ((w_667_3 i), ((w_667_4 i), ((w_667_5 i), ()))))))))
    (h668_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_668 Backend.backend model params i (w_667_0, (w_667_1, (w_667_2, (w_667_3, (w_667_4, (w_667_5, ())))))) ((w_668_0 i), ((w_668_1 i), ((w_668_2 i), ((w_668_3 i), ((w_668_4 i), ((w_668_5 i), ()))))))))
    (h669_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_669 Backend.backend model params i (w_641_0, (w_642_0, (w_643_0, (w_644_0, (w_645_0, (w_646_0, (w_647_0, ((w_668_4 i), ((w_668_0 i), ((w_668_2 i), ((w_668_5 i), ((w_668_1 i), ((w_668_3 i), ()))))))))))))) ((w_669_0 i), ((w_669_1 i), ((w_669_2 i), ((w_669_3 i), ((w_669_4 i), ((w_669_5 i), ()))))))))
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
    (h688_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => Stage_online_eval_0_0.sequential_generatedRoot_688 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_669_0, (w_669_1, (w_669_2, (w_669_3, (w_669_4, (w_669_5, ())))))))))))) next) (Int.toNat (158)) (w_672_0, (w_675_0, (w_678_0, (w_681_0, (w_684_0, (w_687_0, ())))))) (w_688_0, (w_688_1, (w_688_2, (w_688_3, (w_688_4, (w_688_5, ())))))))
    {pw_407_0 : ByteArray}
    {pw_423_0 : Fin 800 → Int}
    {pw_424_0 : Fin 800 → Int}
    {pw_427_0 : Fin 800 → Int}
    {pw_428_0 : Fin 800 → Int}
    {pw_416_0 : ExactMatrix Q6 8 1 1}
    {pw_417_0 : ExactMatrix Q6 8 1 1}
    {pw_418_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_418_1 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_418_2 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_418_3 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_419_0 : ExactMatrix Q6 8 1 10}
    {pw_420_0 : ExactMatrix Q6 8 1 10}
    {pw_421_0 : ExactMatrix Q6 8 1 10}
    {pw_422_0 : ExactMatrix Q6 8 1 10}
    {pw_425_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_425_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_425_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_425_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_425_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_425_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_425_6 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_425_7 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_426_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_426_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_426_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_426_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_429_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_429_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_429_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_429_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_429_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_429_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_429_6 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_429_7 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_431_0 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_431_1 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_431_2 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_431_3 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_432_0 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_432_1 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_432_2 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_432_3 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_433_0 : ExactMatrix Q1 8 1 1}
    {pw_434_0 : ExactMatrix Q1 8 1 1}
    {pw_435_0 : ExactMatrix Q1 8 1 1}
    {pw_436_0 : ExactMatrix Q1 8 1 1}
    {pw_437_0 : ExactMatrix Q1 8 1 1}
    {pw_438_0 : ExactMatrix Q1 8 1 1}
    {pw_439_0 : ExactMatrix Q1 8 1 1}
    {pw_440_0 : ExactMatrix Q1 8 1 1}
    {pw_441_0 : ExactMatrix Q1 8 1 1}
    {pw_442_0 : ExactMatrix Q1 8 1 1}
    {pw_443_0 : ExactMatrix Q1 8 1 1}
    {pw_444_0 : ExactMatrix Q1 8 1 1}
    {pw_445_0 : ExactMatrix Q1 8 1 1}
    {pw_445_1 : ExactMatrix Q1 8 1 1}
    {pw_445_2 : ExactMatrix Q1 8 1 1}
    {pw_445_3 : ExactMatrix Q1 8 1 1}
    (pe_435_0 : pw_435_0 = matrixAdd pw_433_0 pw_434_0)
    (pe_438_0 : pw_438_0 = matrixAdd pw_436_0 pw_437_0)
    (pe_441_0 : pw_441_0 = matrixAdd pw_439_0 pw_440_0)
    (pe_444_0 : pw_444_0 = matrixAdd pw_442_0 pw_443_0)
    (ph425_7 : (∀ i : Fin 800, Stage_preprocess_0_0.parallel_generatedRoot_425 Backend.backend model params' i (pw_407_0, (pw_419_0, (pw_420_0, (pw_421_0, (pw_422_0, (pw_416_0, ((pw_423_0 i), ((pw_424_0 i), (pw_417_0, ()))))))))) ((pw_425_0 i), ((pw_425_1 i), ((pw_425_2 i), ((pw_425_3 i), ((pw_425_4 i), ((pw_425_5 i), ((pw_425_6 i), ((pw_425_7 i), ()))))))))))
    (ph426_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_426 Backend.backend model params' i (pw_425_0, (pw_425_1, (pw_425_2, (pw_425_3, ())))) ((pw_426_0 i), ((pw_426_1 i), ((pw_426_2 i), ((pw_426_3 i), ()))))))
    (ph429_7 : (∀ i : Fin 800, Stage_preprocess_0_0.parallel_generatedRoot_429 Backend.backend model params' i (pw_407_0, (pw_426_0, (pw_426_2, (pw_426_1, (pw_426_3, (pw_416_0, ((pw_427_0 i), ((pw_428_0 i), (pw_417_0, ()))))))))) ((pw_429_0 i), ((pw_429_1 i), ((pw_429_2 i), ((pw_429_3 i), ((pw_429_4 i), ((pw_429_5 i), ((pw_429_6 i), ((pw_429_7 i), ()))))))))))
    (ph431_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_431 Backend.backend model params' i (pw_429_0, (pw_429_1, (pw_429_2, (pw_429_3, ())))) ((pw_431_0 i), ((pw_431_1 i), ((pw_431_2 i), ((pw_431_3 i), ()))))))
    (ph432_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_432 Backend.backend model params' i (pw_407_0, ((pw_431_0 i), ((pw_431_2 i), (pw_418_0, ((pw_431_1 i), ((pw_431_3 i), (pw_418_2, (pw_418_1, (pw_418_3, ()))))))))) ((pw_432_0 i), ((pw_432_1 i), ((pw_432_2 i), ((pw_432_3 i), ()))))))
    (ph433_0 : familyGetStatic pw_432_0 (0) pw_433_0)
    (ph434_0 : familyGetStatic pw_432_0 (1) pw_434_0)
    (ph436_0 : familyGetStatic pw_432_1 (0) pw_436_0)
    (ph437_0 : familyGetStatic pw_432_1 (1) pw_437_0)
    (ph439_0 : familyGetStatic pw_432_2 (0) pw_439_0)
    (ph440_0 : familyGetStatic pw_432_2 (1) pw_440_0)
    (ph442_0 : familyGetStatic pw_432_3 (0) pw_442_0)
    (ph443_0 : familyGetStatic pw_432_3 (1) pw_443_0)
    (ph445_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => Stage_preprocess_0_0.sequential_generatedRoot_445 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_432_0, (pw_432_1, (pw_432_2, (pw_432_3, ())))))))) next) (Int.toNat (158)) (pw_435_0, (pw_438_0, (pw_441_0, (pw_444_0, ())))) (pw_445_0, (pw_445_1, (pw_445_2, (pw_445_3, ())))))
    (hk : w_646_0 = pw_407_0)
    (ho0 : w_654_0 = pw_419_0) (ho1 : w_655_0 = pw_420_0) (ho2 : w_658_0 = pw_421_0) (ho3 : w_659_0 = pw_422_0)
    (hf0 : w_641_0 = pw_418_0) (hf1 : w_642_0 = pw_418_1) (hf2 : w_643_0 = pw_418_2) (hf3 : w_644_0 = pw_418_3)
    (hc0 : w_661_0 = pw_425_6) (hc1 : w_662_0 = pw_425_7)
    (hc2 : w_665_0 = pw_429_6) (hc3 : w_666_0 = pw_429_7)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![w_649_0, w_652_0, w_650_0, w_653_0] c k =
      model.sample Q6 N 1 12 w_646_0 (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)])) :
    w_688_1 = pw_445_1 ∧ w_688_3 = pw_445_3 := by
  subst hk ho0 ho1 ho2 ho3 hf0 hf1 hf2 hf3 hc0 hc1 hc2 hc3
  have hc663 : ∀ i : Fin 800, w_663_0 i = pw_425_0 i ∧ w_663_1 i = pw_425_1 i ∧
      w_663_2 i = pw_425_2 i ∧ w_663_3 i = pw_425_3 i := fun i ↦
    keysCell663 (hon := h663_5 i) (hpre := ph425_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm663 : w_664_0 = pw_426_0 ∧ w_664_1 = pw_426_1 ∧ w_664_2 = pw_426_2 ∧ w_664_3 = pw_426_3 := by
    have a := fun o ↦ members664 (h664_5 o)
    have b := fun o ↦ premembers426 (ph426_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc663 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc663 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc663 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc663 _).2.2.2
  obtain ⟨m0, m1, m2, m3⟩ := hm663
  rw [m0, m1, m2, m3] at h667_5
  have hc667 : ∀ i : Fin 800, w_667_0 i = pw_429_0 i ∧ w_667_1 i = pw_429_1 i ∧
      w_667_2 i = pw_429_2 i ∧ w_667_3 i = pw_429_3 i := fun i ↦
    keysCell667 (hon := h667_5 i) (hpre := ph429_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm667 : w_668_0 = pw_431_0 ∧ w_668_1 = pw_431_1 ∧ w_668_2 = pw_431_2 ∧ w_668_3 = pw_431_3 := by
    have a := fun o ↦ members668 (h668_5 o)
    have b := fun o ↦ premembers431 (ph431_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc667 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc667 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc667 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc667 _).2.2.2
  obtain ⟨n0, n1, n2, n3⟩ := hm667
  rw [n0, n1, n2, n3] at h669_5
  have hl : ∀ o : Fin 160, w_669_0 o = pw_432_0 o ∧ w_669_1 o = pw_432_1 o ∧ w_669_2 o = pw_432_2 o ∧
      w_669_3 o = pw_432_3 o := by
    intro o
    have a := keyRound669 (h669_5 o)
    have b := preKeyRound432 (ph432_3 o)
    rw [show smudgeWeight669 = smudgeWeight560 from rfl] at a
    rw [← b] at a
    simp only [Prod.mk.injEq] at a
    exact ⟨a.1, a.2.1, a.2.2.1, a.2.2.2.1⟩
  subst e_672_0 e_675_0 e_678_0 e_681_0 e_684_0 e_687_0 pe_435_0 pe_438_0 pe_441_0 pe_444_0
  have a := rootSum688 h670_0 h671_0 h673_0 h674_0 h676_0 h677_0 h679_0 h680_0 h682_0 h683_0 h685_0 h686_0 h688_5
  have b := prerootSum445 ph433_0 ph434_0 ph436_0 ph437_0 ph439_0 ph440_0 ph442_0 ph443_0 ph445_3
  simp only [Prod.mk.injEq] at a b
  rw [a.2.1, a.2.2.2.1, b.2.1, b.2.2.2.1]
  exact ⟨Finset.sum_congr rfl fun o _ ↦ (hl o).2.1, Finset.sum_congr rfl fun o _ ↦ (hl o).2.2.2⟩

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The smudge keys of decoder `3`: the online root's are the preprocessing's. -/
theorem smudgeKeyLink3 {model : HashModel} {params : Stage_online_eval_0_0.Params} {params' : Stage_preprocess_0_0.Params}
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
    (h717_5 : (∀ i : Fin 800, Stage_online_eval_0_0.parallel_generatedRoot_717 Backend.backend model params i (w_702_0, (w_703_0, (w_704_0, (w_705_0, (w_706_0, (w_707_0, (w_708_0, (w_709_0, (w_710_0, (w_711_0, (w_712_0, (w_713_0, (w_714_0, (w_700_0, ((w_715_0 i), ((w_716_0 i), ())))))))))))))))) ((w_717_0 i), ((w_717_1 i), ((w_717_2 i), ((w_717_3 i), ((w_717_4 i), ((w_717_5 i), ()))))))))
    (h718_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_718 Backend.backend model params i (w_717_0, (w_717_1, (w_717_2, (w_717_3, (w_717_4, (w_717_5, ())))))) ((w_718_0 i), ((w_718_1 i), ((w_718_2 i), ((w_718_3 i), ((w_718_4 i), ((w_718_5 i), ()))))))))
    (h721_5 : (∀ i : Fin 800, Stage_online_eval_0_0.parallel_generatedRoot_721 Backend.backend model params i (w_702_0, (w_703_0, (w_704_0, (w_705_0, (w_706_0, (w_707_0, (w_718_0, (w_718_2, (w_718_4, (w_711_0, (w_718_1, (w_718_3, (w_718_5, (w_700_0, ((w_719_0 i), ((w_720_0 i), ())))))))))))))))) ((w_721_0 i), ((w_721_1 i), ((w_721_2 i), ((w_721_3 i), ((w_721_4 i), ((w_721_5 i), ()))))))))
    (h722_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_722 Backend.backend model params i (w_721_0, (w_721_1, (w_721_2, (w_721_3, (w_721_4, (w_721_5, ())))))) ((w_722_0 i), ((w_722_1 i), ((w_722_2 i), ((w_722_3 i), ((w_722_4 i), ((w_722_5 i), ()))))))))
    (h723_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_723 Backend.backend model params i (w_695_0, (w_696_0, (w_697_0, (w_698_0, (w_699_0, (w_700_0, (w_701_0, ((w_722_4 i), ((w_722_0 i), ((w_722_2 i), ((w_722_5 i), ((w_722_1 i), ((w_722_3 i), ()))))))))))))) ((w_723_0 i), ((w_723_1 i), ((w_723_2 i), ((w_723_3 i), ((w_723_4 i), ((w_723_5 i), ()))))))))
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
    (h742_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => Stage_online_eval_0_0.sequential_generatedRoot_742 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_723_0, (w_723_1, (w_723_2, (w_723_3, (w_723_4, (w_723_5, ())))))))))))) next) (Int.toNat (158)) (w_726_0, (w_729_0, (w_732_0, (w_735_0, (w_738_0, (w_741_0, ())))))) (w_742_0, (w_742_1, (w_742_2, (w_742_3, (w_742_4, (w_742_5, ())))))))
    {pw_495_0 : ByteArray}
    {pw_511_0 : Fin 800 → Int}
    {pw_512_0 : Fin 800 → Int}
    {pw_515_0 : Fin 800 → Int}
    {pw_516_0 : Fin 800 → Int}
    {pw_504_0 : ExactMatrix Q6 8 1 1}
    {pw_505_0 : ExactMatrix Q6 8 1 1}
    {pw_506_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_506_1 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_506_2 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_506_3 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_507_0 : ExactMatrix Q6 8 1 10}
    {pw_508_0 : ExactMatrix Q6 8 1 10}
    {pw_509_0 : ExactMatrix Q6 8 1 10}
    {pw_510_0 : ExactMatrix Q6 8 1 10}
    {pw_513_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_513_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_513_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_513_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_513_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_513_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_513_6 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_513_7 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_514_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_514_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_514_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_514_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_517_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_517_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_517_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_517_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_517_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_517_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_517_6 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_517_7 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_519_0 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_519_1 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_519_2 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_519_3 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_520_0 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_520_1 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_520_2 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_520_3 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_521_0 : ExactMatrix Q1 8 1 1}
    {pw_522_0 : ExactMatrix Q1 8 1 1}
    {pw_523_0 : ExactMatrix Q1 8 1 1}
    {pw_524_0 : ExactMatrix Q1 8 1 1}
    {pw_525_0 : ExactMatrix Q1 8 1 1}
    {pw_526_0 : ExactMatrix Q1 8 1 1}
    {pw_527_0 : ExactMatrix Q1 8 1 1}
    {pw_528_0 : ExactMatrix Q1 8 1 1}
    {pw_529_0 : ExactMatrix Q1 8 1 1}
    {pw_530_0 : ExactMatrix Q1 8 1 1}
    {pw_531_0 : ExactMatrix Q1 8 1 1}
    {pw_532_0 : ExactMatrix Q1 8 1 1}
    {pw_533_0 : ExactMatrix Q1 8 1 1}
    {pw_533_1 : ExactMatrix Q1 8 1 1}
    {pw_533_2 : ExactMatrix Q1 8 1 1}
    {pw_533_3 : ExactMatrix Q1 8 1 1}
    (pe_523_0 : pw_523_0 = matrixAdd pw_521_0 pw_522_0)
    (pe_526_0 : pw_526_0 = matrixAdd pw_524_0 pw_525_0)
    (pe_529_0 : pw_529_0 = matrixAdd pw_527_0 pw_528_0)
    (pe_532_0 : pw_532_0 = matrixAdd pw_530_0 pw_531_0)
    (ph513_7 : (∀ i : Fin 800, Stage_preprocess_0_0.parallel_generatedRoot_513 Backend.backend model params' i (pw_495_0, (pw_507_0, (pw_508_0, (pw_509_0, (pw_510_0, (pw_504_0, ((pw_511_0 i), ((pw_512_0 i), (pw_505_0, ()))))))))) ((pw_513_0 i), ((pw_513_1 i), ((pw_513_2 i), ((pw_513_3 i), ((pw_513_4 i), ((pw_513_5 i), ((pw_513_6 i), ((pw_513_7 i), ()))))))))))
    (ph514_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_514 Backend.backend model params' i (pw_513_0, (pw_513_1, (pw_513_2, (pw_513_3, ())))) ((pw_514_0 i), ((pw_514_1 i), ((pw_514_2 i), ((pw_514_3 i), ()))))))
    (ph517_7 : (∀ i : Fin 800, Stage_preprocess_0_0.parallel_generatedRoot_517 Backend.backend model params' i (pw_495_0, (pw_514_0, (pw_514_2, (pw_514_1, (pw_514_3, (pw_504_0, ((pw_515_0 i), ((pw_516_0 i), (pw_505_0, ()))))))))) ((pw_517_0 i), ((pw_517_1 i), ((pw_517_2 i), ((pw_517_3 i), ((pw_517_4 i), ((pw_517_5 i), ((pw_517_6 i), ((pw_517_7 i), ()))))))))))
    (ph519_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_519 Backend.backend model params' i (pw_517_0, (pw_517_1, (pw_517_2, (pw_517_3, ())))) ((pw_519_0 i), ((pw_519_1 i), ((pw_519_2 i), ((pw_519_3 i), ()))))))
    (ph520_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_520 Backend.backend model params' i (pw_495_0, ((pw_519_0 i), ((pw_519_2 i), (pw_506_0, ((pw_519_1 i), ((pw_519_3 i), (pw_506_2, (pw_506_1, (pw_506_3, ()))))))))) ((pw_520_0 i), ((pw_520_1 i), ((pw_520_2 i), ((pw_520_3 i), ()))))))
    (ph521_0 : familyGetStatic pw_520_0 (0) pw_521_0)
    (ph522_0 : familyGetStatic pw_520_0 (1) pw_522_0)
    (ph524_0 : familyGetStatic pw_520_1 (0) pw_524_0)
    (ph525_0 : familyGetStatic pw_520_1 (1) pw_525_0)
    (ph527_0 : familyGetStatic pw_520_2 (0) pw_527_0)
    (ph528_0 : familyGetStatic pw_520_2 (1) pw_528_0)
    (ph530_0 : familyGetStatic pw_520_3 (0) pw_530_0)
    (ph531_0 : familyGetStatic pw_520_3 (1) pw_531_0)
    (ph533_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => Stage_preprocess_0_0.sequential_generatedRoot_533 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_520_0, (pw_520_1, (pw_520_2, (pw_520_3, ())))))))) next) (Int.toNat (158)) (pw_523_0, (pw_526_0, (pw_529_0, (pw_532_0, ())))) (pw_533_0, (pw_533_1, (pw_533_2, (pw_533_3, ())))))
    (hk : w_700_0 = pw_495_0)
    (ho0 : w_708_0 = pw_507_0) (ho1 : w_709_0 = pw_508_0) (ho2 : w_712_0 = pw_509_0) (ho3 : w_713_0 = pw_510_0)
    (hf0 : w_695_0 = pw_506_0) (hf1 : w_696_0 = pw_506_1) (hf2 : w_697_0 = pw_506_2) (hf3 : w_698_0 = pw_506_3)
    (hc0 : w_715_0 = pw_513_6) (hc1 : w_716_0 = pw_513_7)
    (hc2 : w_719_0 = pw_517_6) (hc3 : w_720_0 = pw_517_7)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![w_703_0, w_706_0, w_704_0, w_707_0] c k =
      model.sample Q6 N 1 12 w_700_0 (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)])) :
    w_742_1 = pw_533_1 ∧ w_742_3 = pw_533_3 := by
  subst hk ho0 ho1 ho2 ho3 hf0 hf1 hf2 hf3 hc0 hc1 hc2 hc3
  have hc717 : ∀ i : Fin 800, w_717_0 i = pw_513_0 i ∧ w_717_1 i = pw_513_1 i ∧
      w_717_2 i = pw_513_2 i ∧ w_717_3 i = pw_513_3 i := fun i ↦
    keysCell717 (hon := h717_5 i) (hpre := ph513_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm717 : w_718_0 = pw_514_0 ∧ w_718_1 = pw_514_1 ∧ w_718_2 = pw_514_2 ∧ w_718_3 = pw_514_3 := by
    have a := fun o ↦ members718 (h718_5 o)
    have b := fun o ↦ premembers514 (ph514_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc717 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc717 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc717 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc717 _).2.2.2
  obtain ⟨m0, m1, m2, m3⟩ := hm717
  rw [m0, m1, m2, m3] at h721_5
  have hc721 : ∀ i : Fin 800, w_721_0 i = pw_517_0 i ∧ w_721_1 i = pw_517_1 i ∧
      w_721_2 i = pw_517_2 i ∧ w_721_3 i = pw_517_3 i := fun i ↦
    keysCell721 (hon := h721_5 i) (hpre := ph517_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm721 : w_722_0 = pw_519_0 ∧ w_722_1 = pw_519_1 ∧ w_722_2 = pw_519_2 ∧ w_722_3 = pw_519_3 := by
    have a := fun o ↦ members722 (h722_5 o)
    have b := fun o ↦ premembers519 (ph519_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc721 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc721 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc721 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc721 _).2.2.2
  obtain ⟨n0, n1, n2, n3⟩ := hm721
  rw [n0, n1, n2, n3] at h723_5
  have hl : ∀ o : Fin 160, w_723_0 o = pw_520_0 o ∧ w_723_1 o = pw_520_1 o ∧ w_723_2 o = pw_520_2 o ∧
      w_723_3 o = pw_520_3 o := by
    intro o
    have a := keyRound723 (h723_5 o)
    have b := preKeyRound520 (ph520_3 o)
    rw [show smudgeWeight723 = smudgeWeight560 from rfl] at a
    rw [← b] at a
    simp only [Prod.mk.injEq] at a
    exact ⟨a.1, a.2.1, a.2.2.1, a.2.2.2.1⟩
  subst e_726_0 e_729_0 e_732_0 e_735_0 e_738_0 e_741_0 pe_523_0 pe_526_0 pe_529_0 pe_532_0
  have a := rootSum742 h724_0 h725_0 h727_0 h728_0 h730_0 h731_0 h733_0 h734_0 h736_0 h737_0 h739_0 h740_0 h742_5
  have b := prerootSum533 ph521_0 ph522_0 ph524_0 ph525_0 ph527_0 ph528_0 ph530_0 ph531_0 ph533_3
  simp only [Prod.mk.injEq] at a b
  rw [a.2.1, a.2.2.2.1, b.2.1, b.2.2.2.1]
  exact ⟨Finset.sum_congr rfl fun o _ ↦ (hl o).2.1, Finset.sum_congr rfl fun o _ ↦ (hl o).2.2.2⟩

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The smudge keys of decoder `4`: the online root's are the preprocessing's. -/
theorem smudgeKeyLink4 {model : HashModel} {params : Stage_online_eval_0_0.Params} {params' : Stage_preprocess_0_0.Params}
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
    (h771_5 : (∀ i : Fin 800, Stage_online_eval_0_0.parallel_generatedRoot_771 Backend.backend model params i (w_756_0, (w_757_0, (w_758_0, (w_759_0, (w_760_0, (w_761_0, (w_762_0, (w_763_0, (w_764_0, (w_765_0, (w_766_0, (w_767_0, (w_768_0, (w_754_0, ((w_769_0 i), ((w_770_0 i), ())))))))))))))))) ((w_771_0 i), ((w_771_1 i), ((w_771_2 i), ((w_771_3 i), ((w_771_4 i), ((w_771_5 i), ()))))))))
    (h772_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_772 Backend.backend model params i (w_771_0, (w_771_1, (w_771_2, (w_771_3, (w_771_4, (w_771_5, ())))))) ((w_772_0 i), ((w_772_1 i), ((w_772_2 i), ((w_772_3 i), ((w_772_4 i), ((w_772_5 i), ()))))))))
    (h775_5 : (∀ i : Fin 800, Stage_online_eval_0_0.parallel_generatedRoot_775 Backend.backend model params i (w_756_0, (w_757_0, (w_758_0, (w_759_0, (w_760_0, (w_761_0, (w_772_0, (w_772_2, (w_772_4, (w_765_0, (w_772_1, (w_772_3, (w_772_5, (w_754_0, ((w_773_0 i), ((w_774_0 i), ())))))))))))))))) ((w_775_0 i), ((w_775_1 i), ((w_775_2 i), ((w_775_3 i), ((w_775_4 i), ((w_775_5 i), ()))))))))
    (h776_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_776 Backend.backend model params i (w_775_0, (w_775_1, (w_775_2, (w_775_3, (w_775_4, (w_775_5, ())))))) ((w_776_0 i), ((w_776_1 i), ((w_776_2 i), ((w_776_3 i), ((w_776_4 i), ((w_776_5 i), ()))))))))
    (h777_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_777 Backend.backend model params i (w_749_0, (w_750_0, (w_751_0, (w_752_0, (w_753_0, (w_754_0, (w_755_0, ((w_776_4 i), ((w_776_0 i), ((w_776_2 i), ((w_776_5 i), ((w_776_1 i), ((w_776_3 i), ()))))))))))))) ((w_777_0 i), ((w_777_1 i), ((w_777_2 i), ((w_777_3 i), ((w_777_4 i), ((w_777_5 i), ()))))))))
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
    (h796_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => Stage_online_eval_0_0.sequential_generatedRoot_796 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_777_0, (w_777_1, (w_777_2, (w_777_3, (w_777_4, (w_777_5, ())))))))))))) next) (Int.toNat (158)) (w_780_0, (w_783_0, (w_786_0, (w_789_0, (w_792_0, (w_795_0, ())))))) (w_796_0, (w_796_1, (w_796_2, (w_796_3, (w_796_4, (w_796_5, ())))))))
    {pw_451_0 : ByteArray}
    {pw_467_0 : Fin 800 → Int}
    {pw_468_0 : Fin 800 → Int}
    {pw_471_0 : Fin 800 → Int}
    {pw_472_0 : Fin 800 → Int}
    {pw_460_0 : ExactMatrix Q6 8 1 1}
    {pw_461_0 : ExactMatrix Q6 8 1 1}
    {pw_462_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_462_1 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_462_2 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_462_3 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_463_0 : ExactMatrix Q6 8 1 10}
    {pw_464_0 : ExactMatrix Q6 8 1 10}
    {pw_465_0 : ExactMatrix Q6 8 1 10}
    {pw_466_0 : ExactMatrix Q6 8 1 10}
    {pw_469_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_469_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_469_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_469_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_469_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_469_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_469_6 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_469_7 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_470_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_470_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_470_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_470_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_473_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_473_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_473_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_473_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_473_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_473_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_473_6 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_473_7 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_475_0 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_475_1 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_475_2 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_475_3 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_476_0 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_476_1 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_476_2 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_476_3 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_477_0 : ExactMatrix Q1 8 1 1}
    {pw_478_0 : ExactMatrix Q1 8 1 1}
    {pw_479_0 : ExactMatrix Q1 8 1 1}
    {pw_480_0 : ExactMatrix Q1 8 1 1}
    {pw_481_0 : ExactMatrix Q1 8 1 1}
    {pw_482_0 : ExactMatrix Q1 8 1 1}
    {pw_483_0 : ExactMatrix Q1 8 1 1}
    {pw_484_0 : ExactMatrix Q1 8 1 1}
    {pw_485_0 : ExactMatrix Q1 8 1 1}
    {pw_486_0 : ExactMatrix Q1 8 1 1}
    {pw_487_0 : ExactMatrix Q1 8 1 1}
    {pw_488_0 : ExactMatrix Q1 8 1 1}
    {pw_489_0 : ExactMatrix Q1 8 1 1}
    {pw_489_1 : ExactMatrix Q1 8 1 1}
    {pw_489_2 : ExactMatrix Q1 8 1 1}
    {pw_489_3 : ExactMatrix Q1 8 1 1}
    (pe_479_0 : pw_479_0 = matrixAdd pw_477_0 pw_478_0)
    (pe_482_0 : pw_482_0 = matrixAdd pw_480_0 pw_481_0)
    (pe_485_0 : pw_485_0 = matrixAdd pw_483_0 pw_484_0)
    (pe_488_0 : pw_488_0 = matrixAdd pw_486_0 pw_487_0)
    (ph469_7 : (∀ i : Fin 800, Stage_preprocess_0_0.parallel_generatedRoot_469 Backend.backend model params' i (pw_451_0, (pw_463_0, (pw_464_0, (pw_465_0, (pw_466_0, (pw_460_0, ((pw_467_0 i), ((pw_468_0 i), (pw_461_0, ()))))))))) ((pw_469_0 i), ((pw_469_1 i), ((pw_469_2 i), ((pw_469_3 i), ((pw_469_4 i), ((pw_469_5 i), ((pw_469_6 i), ((pw_469_7 i), ()))))))))))
    (ph470_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_470 Backend.backend model params' i (pw_469_0, (pw_469_1, (pw_469_2, (pw_469_3, ())))) ((pw_470_0 i), ((pw_470_1 i), ((pw_470_2 i), ((pw_470_3 i), ()))))))
    (ph473_7 : (∀ i : Fin 800, Stage_preprocess_0_0.parallel_generatedRoot_473 Backend.backend model params' i (pw_451_0, (pw_470_0, (pw_470_2, (pw_470_1, (pw_470_3, (pw_460_0, ((pw_471_0 i), ((pw_472_0 i), (pw_461_0, ()))))))))) ((pw_473_0 i), ((pw_473_1 i), ((pw_473_2 i), ((pw_473_3 i), ((pw_473_4 i), ((pw_473_5 i), ((pw_473_6 i), ((pw_473_7 i), ()))))))))))
    (ph475_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_475 Backend.backend model params' i (pw_473_0, (pw_473_1, (pw_473_2, (pw_473_3, ())))) ((pw_475_0 i), ((pw_475_1 i), ((pw_475_2 i), ((pw_475_3 i), ()))))))
    (ph476_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_476 Backend.backend model params' i (pw_451_0, ((pw_475_0 i), ((pw_475_2 i), (pw_462_0, ((pw_475_1 i), ((pw_475_3 i), (pw_462_2, (pw_462_1, (pw_462_3, ()))))))))) ((pw_476_0 i), ((pw_476_1 i), ((pw_476_2 i), ((pw_476_3 i), ()))))))
    (ph477_0 : familyGetStatic pw_476_0 (0) pw_477_0)
    (ph478_0 : familyGetStatic pw_476_0 (1) pw_478_0)
    (ph480_0 : familyGetStatic pw_476_1 (0) pw_480_0)
    (ph481_0 : familyGetStatic pw_476_1 (1) pw_481_0)
    (ph483_0 : familyGetStatic pw_476_2 (0) pw_483_0)
    (ph484_0 : familyGetStatic pw_476_2 (1) pw_484_0)
    (ph486_0 : familyGetStatic pw_476_3 (0) pw_486_0)
    (ph487_0 : familyGetStatic pw_476_3 (1) pw_487_0)
    (ph489_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => Stage_preprocess_0_0.sequential_generatedRoot_489 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_476_0, (pw_476_1, (pw_476_2, (pw_476_3, ())))))))) next) (Int.toNat (158)) (pw_479_0, (pw_482_0, (pw_485_0, (pw_488_0, ())))) (pw_489_0, (pw_489_1, (pw_489_2, (pw_489_3, ())))))
    (hk : w_754_0 = pw_451_0)
    (ho0 : w_762_0 = pw_463_0) (ho1 : w_763_0 = pw_464_0) (ho2 : w_766_0 = pw_465_0) (ho3 : w_767_0 = pw_466_0)
    (hf0 : w_749_0 = pw_462_0) (hf1 : w_750_0 = pw_462_1) (hf2 : w_751_0 = pw_462_2) (hf3 : w_752_0 = pw_462_3)
    (hc0 : w_769_0 = pw_469_6) (hc1 : w_770_0 = pw_469_7)
    (hc2 : w_773_0 = pw_473_6) (hc3 : w_774_0 = pw_473_7)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![w_757_0, w_760_0, w_758_0, w_761_0] c k =
      model.sample Q6 N 1 12 w_754_0 (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)])) :
    w_796_1 = pw_489_1 ∧ w_796_3 = pw_489_3 := by
  subst hk ho0 ho1 ho2 ho3 hf0 hf1 hf2 hf3 hc0 hc1 hc2 hc3
  have hc771 : ∀ i : Fin 800, w_771_0 i = pw_469_0 i ∧ w_771_1 i = pw_469_1 i ∧
      w_771_2 i = pw_469_2 i ∧ w_771_3 i = pw_469_3 i := fun i ↦
    keysCell771 (hon := h771_5 i) (hpre := ph469_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm771 : w_772_0 = pw_470_0 ∧ w_772_1 = pw_470_1 ∧ w_772_2 = pw_470_2 ∧ w_772_3 = pw_470_3 := by
    have a := fun o ↦ members772 (h772_5 o)
    have b := fun o ↦ premembers470 (ph470_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc771 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc771 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc771 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc771 _).2.2.2
  obtain ⟨m0, m1, m2, m3⟩ := hm771
  rw [m0, m1, m2, m3] at h775_5
  have hc775 : ∀ i : Fin 800, w_775_0 i = pw_473_0 i ∧ w_775_1 i = pw_473_1 i ∧
      w_775_2 i = pw_473_2 i ∧ w_775_3 i = pw_473_3 i := fun i ↦
    keysCell775 (hon := h775_5 i) (hpre := ph473_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm775 : w_776_0 = pw_475_0 ∧ w_776_1 = pw_475_1 ∧ w_776_2 = pw_475_2 ∧ w_776_3 = pw_475_3 := by
    have a := fun o ↦ members776 (h776_5 o)
    have b := fun o ↦ premembers475 (ph475_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc775 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc775 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc775 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc775 _).2.2.2
  obtain ⟨n0, n1, n2, n3⟩ := hm775
  rw [n0, n1, n2, n3] at h777_5
  have hl : ∀ o : Fin 160, w_777_0 o = pw_476_0 o ∧ w_777_1 o = pw_476_1 o ∧ w_777_2 o = pw_476_2 o ∧
      w_777_3 o = pw_476_3 o := by
    intro o
    have a := keyRound777 (h777_5 o)
    have b := preKeyRound476 (ph476_3 o)
    rw [show smudgeWeight777 = smudgeWeight560 from rfl] at a
    rw [← b] at a
    simp only [Prod.mk.injEq] at a
    exact ⟨a.1, a.2.1, a.2.2.1, a.2.2.2.1⟩
  subst e_780_0 e_783_0 e_786_0 e_789_0 e_792_0 e_795_0 pe_479_0 pe_482_0 pe_485_0 pe_488_0
  have a := rootSum796 h778_0 h779_0 h781_0 h782_0 h784_0 h785_0 h787_0 h788_0 h790_0 h791_0 h793_0 h794_0 h796_5
  have b := prerootSum489 ph477_0 ph478_0 ph480_0 ph481_0 ph483_0 ph484_0 ph486_0 ph487_0 ph489_3
  simp only [Prod.mk.injEq] at a b
  rw [a.2.1, a.2.2.2.1, b.2.1, b.2.2.2.1]
  exact ⟨Finset.sum_congr rfl fun o _ ↦ (hl o).2.1, Finset.sum_congr rfl fun o _ ↦ (hl o).2.2.2⟩

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The smudge keys of decoder `5`: the online root's are the preprocessing's. -/
theorem smudgeKeyLink5 {model : HashModel} {params : Stage_online_eval_0_0.Params} {params' : Stage_preprocess_0_0.Params}
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
    (h825_5 : (∀ i : Fin 800, Stage_online_eval_0_0.parallel_generatedRoot_825 Backend.backend model params i (w_810_0, (w_811_0, (w_812_0, (w_813_0, (w_814_0, (w_815_0, (w_816_0, (w_817_0, (w_818_0, (w_819_0, (w_820_0, (w_821_0, (w_822_0, (w_808_0, ((w_823_0 i), ((w_824_0 i), ())))))))))))))))) ((w_825_0 i), ((w_825_1 i), ((w_825_2 i), ((w_825_3 i), ((w_825_4 i), ((w_825_5 i), ()))))))))
    (h826_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_826 Backend.backend model params i (w_825_0, (w_825_1, (w_825_2, (w_825_3, (w_825_4, (w_825_5, ())))))) ((w_826_0 i), ((w_826_1 i), ((w_826_2 i), ((w_826_3 i), ((w_826_4 i), ((w_826_5 i), ()))))))))
    (h829_5 : (∀ i : Fin 800, Stage_online_eval_0_0.parallel_generatedRoot_829 Backend.backend model params i (w_810_0, (w_811_0, (w_812_0, (w_813_0, (w_814_0, (w_815_0, (w_826_0, (w_826_2, (w_826_4, (w_819_0, (w_826_1, (w_826_3, (w_826_5, (w_808_0, ((w_827_0 i), ((w_828_0 i), ())))))))))))))))) ((w_829_0 i), ((w_829_1 i), ((w_829_2 i), ((w_829_3 i), ((w_829_4 i), ((w_829_5 i), ()))))))))
    (h830_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_830 Backend.backend model params i (w_829_0, (w_829_1, (w_829_2, (w_829_3, (w_829_4, (w_829_5, ())))))) ((w_830_0 i), ((w_830_1 i), ((w_830_2 i), ((w_830_3 i), ((w_830_4 i), ((w_830_5 i), ()))))))))
    (h831_5 : (∀ i : Fin 160, Stage_online_eval_0_0.parallel_generatedRoot_831 Backend.backend model params i (w_803_0, (w_804_0, (w_805_0, (w_806_0, (w_807_0, (w_808_0, (w_809_0, ((w_830_4 i), ((w_830_0 i), ((w_830_2 i), ((w_830_5 i), ((w_830_1 i), ((w_830_3 i), ()))))))))))))) ((w_831_0 i), ((w_831_1 i), ((w_831_2 i), ((w_831_3 i), ((w_831_4 i), ((w_831_5 i), ()))))))))
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
    (h850_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => Stage_online_eval_0_0.sequential_generatedRoot_850 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_831_0, (w_831_1, (w_831_2, (w_831_3, (w_831_4, (w_831_5, ())))))))))))) next) (Int.toNat (158)) (w_834_0, (w_837_0, (w_840_0, (w_843_0, (w_846_0, (w_849_0, ())))))) (w_850_0, (w_850_1, (w_850_2, (w_850_3, (w_850_4, (w_850_5, ())))))))
    {pw_539_0 : ByteArray}
    {pw_555_0 : Fin 800 → Int}
    {pw_556_0 : Fin 800 → Int}
    {pw_559_0 : Fin 800 → Int}
    {pw_560_0 : Fin 800 → Int}
    {pw_548_0 : ExactMatrix Q6 8 1 1}
    {pw_549_0 : ExactMatrix Q6 8 1 1}
    {pw_550_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_550_1 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_550_2 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_550_3 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_551_0 : ExactMatrix Q6 8 1 10}
    {pw_552_0 : ExactMatrix Q6 8 1 10}
    {pw_553_0 : ExactMatrix Q6 8 1 10}
    {pw_554_0 : ExactMatrix Q6 8 1 10}
    {pw_557_0 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_557_1 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_557_2 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_557_3 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_557_4 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_557_5 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_557_6 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_557_7 : Fin 800 → ExactMatrix Q5 8 1 8}
    {pw_558_0 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_558_1 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_558_2 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_558_3 : Fin 160 → ExactMatrix Q5 8 1 8}
    {pw_561_0 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_561_1 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_561_2 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_561_3 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_561_4 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_561_5 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_561_6 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_561_7 : Fin 800 → ExactMatrix Q0 8 1 6}
    {pw_563_0 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_563_1 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_563_2 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_563_3 : Fin 160 → ExactMatrix Q0 8 1 6}
    {pw_564_0 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_564_1 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_564_2 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_564_3 : Fin 160 → ExactMatrix Q1 8 1 1}
    {pw_565_0 : ExactMatrix Q1 8 1 1}
    {pw_566_0 : ExactMatrix Q1 8 1 1}
    {pw_567_0 : ExactMatrix Q1 8 1 1}
    {pw_568_0 : ExactMatrix Q1 8 1 1}
    {pw_569_0 : ExactMatrix Q1 8 1 1}
    {pw_570_0 : ExactMatrix Q1 8 1 1}
    {pw_571_0 : ExactMatrix Q1 8 1 1}
    {pw_572_0 : ExactMatrix Q1 8 1 1}
    {pw_573_0 : ExactMatrix Q1 8 1 1}
    {pw_574_0 : ExactMatrix Q1 8 1 1}
    {pw_575_0 : ExactMatrix Q1 8 1 1}
    {pw_576_0 : ExactMatrix Q1 8 1 1}
    {pw_577_0 : ExactMatrix Q1 8 1 1}
    {pw_577_1 : ExactMatrix Q1 8 1 1}
    {pw_577_2 : ExactMatrix Q1 8 1 1}
    {pw_577_3 : ExactMatrix Q1 8 1 1}
    (pe_567_0 : pw_567_0 = matrixAdd pw_565_0 pw_566_0)
    (pe_570_0 : pw_570_0 = matrixAdd pw_568_0 pw_569_0)
    (pe_573_0 : pw_573_0 = matrixAdd pw_571_0 pw_572_0)
    (pe_576_0 : pw_576_0 = matrixAdd pw_574_0 pw_575_0)
    (ph557_7 : (∀ i : Fin 800, Stage_preprocess_0_0.parallel_generatedRoot_557 Backend.backend model params' i (pw_539_0, (pw_551_0, (pw_552_0, (pw_553_0, (pw_554_0, (pw_548_0, ((pw_555_0 i), ((pw_556_0 i), (pw_549_0, ()))))))))) ((pw_557_0 i), ((pw_557_1 i), ((pw_557_2 i), ((pw_557_3 i), ((pw_557_4 i), ((pw_557_5 i), ((pw_557_6 i), ((pw_557_7 i), ()))))))))))
    (ph558_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_558 Backend.backend model params' i (pw_557_0, (pw_557_1, (pw_557_2, (pw_557_3, ())))) ((pw_558_0 i), ((pw_558_1 i), ((pw_558_2 i), ((pw_558_3 i), ()))))))
    (ph561_7 : (∀ i : Fin 800, Stage_preprocess_0_0.parallel_generatedRoot_561 Backend.backend model params' i (pw_539_0, (pw_558_0, (pw_558_2, (pw_558_1, (pw_558_3, (pw_548_0, ((pw_559_0 i), ((pw_560_0 i), (pw_549_0, ()))))))))) ((pw_561_0 i), ((pw_561_1 i), ((pw_561_2 i), ((pw_561_3 i), ((pw_561_4 i), ((pw_561_5 i), ((pw_561_6 i), ((pw_561_7 i), ()))))))))))
    (ph563_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_563 Backend.backend model params' i (pw_561_0, (pw_561_1, (pw_561_2, (pw_561_3, ())))) ((pw_563_0 i), ((pw_563_1 i), ((pw_563_2 i), ((pw_563_3 i), ()))))))
    (ph564_3 : (∀ i : Fin 160, Stage_preprocess_0_0.parallel_generatedRoot_564 Backend.backend model params' i (pw_539_0, ((pw_563_0 i), ((pw_563_2 i), (pw_550_0, ((pw_563_1 i), ((pw_563_3 i), (pw_550_2, (pw_550_1, (pw_550_3, ()))))))))) ((pw_564_0 i), ((pw_564_1 i), ((pw_564_2 i), ((pw_564_3 i), ()))))))
    (ph565_0 : familyGetStatic pw_564_0 (0) pw_565_0)
    (ph566_0 : familyGetStatic pw_564_0 (1) pw_566_0)
    (ph568_0 : familyGetStatic pw_564_1 (0) pw_568_0)
    (ph569_0 : familyGetStatic pw_564_1 (1) pw_569_0)
    (ph571_0 : familyGetStatic pw_564_2 (0) pw_571_0)
    (ph572_0 : familyGetStatic pw_564_2 (1) pw_572_0)
    (ph574_0 : familyGetStatic pw_564_3 (0) pw_574_0)
    (ph575_0 : familyGetStatic pw_564_3 (1) pw_575_0)
    (ph577_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × ExactMatrix Q1 8 1 1 × Unit) => Stage_preprocess_0_0.sequential_generatedRoot_577 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_564_0, (pw_564_1, (pw_564_2, (pw_564_3, ())))))))) next) (Int.toNat (158)) (pw_567_0, (pw_570_0, (pw_573_0, (pw_576_0, ())))) (pw_577_0, (pw_577_1, (pw_577_2, (pw_577_3, ())))))
    (hk : w_808_0 = pw_539_0)
    (ho0 : w_816_0 = pw_551_0) (ho1 : w_817_0 = pw_552_0) (ho2 : w_820_0 = pw_553_0) (ho3 : w_821_0 = pw_554_0)
    (hf0 : w_803_0 = pw_550_0) (hf1 : w_804_0 = pw_550_1) (hf2 : w_805_0 = pw_550_2) (hf3 : w_806_0 = pw_550_3)
    (hc0 : w_823_0 = pw_557_6) (hc1 : w_824_0 = pw_557_7)
    (hc2 : w_827_0 = pw_561_6) (hc3 : w_828_0 = pw_561_7)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![w_811_0, w_814_0, w_812_0, w_815_0] c k =
      model.sample Q6 N 1 12 w_808_0 (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)])) :
    w_850_1 = pw_577_1 ∧ w_850_3 = pw_577_3 := by
  subst hk ho0 ho1 ho2 ho3 hf0 hf1 hf2 hf3 hc0 hc1 hc2 hc3
  have hc825 : ∀ i : Fin 800, w_825_0 i = pw_557_0 i ∧ w_825_1 i = pw_557_1 i ∧
      w_825_2 i = pw_557_2 i ∧ w_825_3 i = pw_557_3 i := fun i ↦
    keysCell825 (hon := h825_5 i) (hpre := ph557_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm825 : w_826_0 = pw_558_0 ∧ w_826_1 = pw_558_1 ∧ w_826_2 = pw_558_2 ∧ w_826_3 = pw_558_3 := by
    have a := fun o ↦ members826 (h826_5 o)
    have b := fun o ↦ premembers558 (ph558_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc825 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc825 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc825 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc825 _).2.2.2
  obtain ⟨m0, m1, m2, m3⟩ := hm825
  rw [m0, m1, m2, m3] at h829_5
  have hc829 : ∀ i : Fin 800, w_829_0 i = pw_561_0 i ∧ w_829_1 i = pw_561_1 i ∧
      w_829_2 i = pw_561_2 i ∧ w_829_3 i = pw_561_3 i := fun i ↦
    keysCell829 (hon := h829_5 i) (hpre := ph561_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm829 : w_830_0 = pw_563_0 ∧ w_830_1 = pw_563_1 ∧ w_830_2 = pw_563_2 ∧ w_830_3 = pw_563_3 := by
    have a := fun o ↦ members830 (h830_5 o)
    have b := fun o ↦ premembers563 (ph563_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc829 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc829 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc829 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc829 _).2.2.2
  obtain ⟨n0, n1, n2, n3⟩ := hm829
  rw [n0, n1, n2, n3] at h831_5
  have hl : ∀ o : Fin 160, w_831_0 o = pw_564_0 o ∧ w_831_1 o = pw_564_1 o ∧ w_831_2 o = pw_564_2 o ∧
      w_831_3 o = pw_564_3 o := by
    intro o
    have a := keyRound831 (h831_5 o)
    have b := preKeyRound564 (ph564_3 o)
    rw [show smudgeWeight831 = smudgeWeight560 from rfl] at a
    rw [← b] at a
    simp only [Prod.mk.injEq] at a
    exact ⟨a.1, a.2.1, a.2.2.1, a.2.2.2.1⟩
  subst e_834_0 e_837_0 e_840_0 e_843_0 e_846_0 e_849_0 pe_567_0 pe_570_0 pe_573_0 pe_576_0
  have a := rootSum850 h832_0 h833_0 h835_0 h836_0 h838_0 h839_0 h841_0 h842_0 h844_0 h845_0 h847_0 h848_0 h850_5
  have b := prerootSum577 ph565_0 ph566_0 ph568_0 ph569_0 ph571_0 ph572_0 ph574_0 ph575_0 ph577_3
  simp only [Prod.mk.injEq] at a b
  rw [a.2.1, a.2.2.2.1, b.2.1, b.2.2.2.1]
  exact ⟨Finset.sum_congr rfl fun o _ ↦ (hl o).2.1, Finset.sum_congr rfl fun o _ ↦ (hl o).2.2.2⟩

end RefreshTopProof.Part_0_0
