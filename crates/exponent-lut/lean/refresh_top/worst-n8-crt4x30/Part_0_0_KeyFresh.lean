import Part_0_0_KeysLuts
import Part_0_0_KeysCells
import Part_0_0_Sums
import Part_0_0_PreSums
import Part_0_0_PrePlace
import Part_0_0_PreRootSums
import Part_0_0_RootSums
import Part_0_0_FreshCarrier

/-! The fresh columns' keys of the online root of part `0_0` are those the preprocessing computes
(written by `generator/gen_key_links.py`). -/

namespace RefreshTopProof.Part_0_0

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The sampled members of the fresh PRF: the online root's keys are the preprocessing's. -/
theorem freshKeyCore {model : HashModel} {params : Stage_online_eval_0_0.Params} {params' : Stage_preprocess_0_0.Params}
    {w_358_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_359_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_360_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_361_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {w_362_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_363_0 : ByteArray}
    {w_364_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_365_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_366_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_367_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_368_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {w_369_0 : Fin 4 → ExactMatrix Q5 8 1 8}
    {w_370_0 : Fin 4 → ExactMatrix Q5 8 1 8}
    {w_371_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {w_372_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {w_373_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {w_374_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {w_375_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {w_376_0 : Fin 5 → ExactMatrix Q7 8 1 14}
    {w_377_0 : ExactMatrix Q7 8 1 12}
    {w_378_0 : ExactMatrix Q7 8 1 12}
    {w_379_0 : ExactMatrix Q7 8 1 12}
    {w_381_0 : ExactMatrix Q7 8 1 12}
    {w_382_0 : ExactMatrix Q7 8 1 12}
    {w_383_0 : ExactMatrix Q7 8 1 12}
    {w_384_0 : Fin 240 → ExactMatrix Q6 8 1 10}
    {w_385_0 : Fin 240 → ExactMatrix Q6 8 1 10}
    {w_388_0 : Fin 240 → ExactMatrix Q5 8 1 8}
    {w_389_0 : Fin 240 → ExactMatrix Q5 8 1 8}
    {w_380_0 : Fin 5 → ExactMatrix Q7 8 1 1}
    {w_386_0 : Fin 240 → ExactMatrix Q6 8 1 10}
    {w_386_1 : Fin 240 → ExactMatrix Q6 8 1 10}
    {w_386_2 : Fin 240 → ExactMatrix Q6 8 1 10}
    {w_386_3 : Fin 240 → ExactMatrix Q6 8 1 10}
    {w_386_4 : Fin 240 → ExactMatrix Q6 8 1 10}
    {w_386_5 : Fin 240 → ExactMatrix Q6 8 1 10}
    {w_387_0 : Fin 48 → ExactMatrix Q6 8 1 10}
    {w_387_1 : Fin 48 → ExactMatrix Q6 8 1 10}
    {w_387_2 : Fin 48 → ExactMatrix Q6 8 1 10}
    {w_387_3 : Fin 48 → ExactMatrix Q6 8 1 10}
    {w_387_4 : Fin 48 → ExactMatrix Q6 8 1 10}
    {w_387_5 : Fin 48 → ExactMatrix Q6 8 1 10}
    {w_390_0 : Fin 240 → ExactMatrix Q5 8 1 8}
    {w_390_1 : Fin 240 → ExactMatrix Q5 8 1 8}
    {w_390_2 : Fin 240 → ExactMatrix Q5 8 1 8}
    {w_390_3 : Fin 240 → ExactMatrix Q5 8 1 8}
    {w_390_4 : Fin 240 → ExactMatrix Q5 8 1 8}
    {w_390_5 : Fin 240 → ExactMatrix Q5 8 1 8}
    {w_391_0 : Fin 48 → ExactMatrix Q5 8 1 8}
    {w_391_1 : Fin 48 → ExactMatrix Q5 8 1 8}
    {w_391_2 : Fin 48 → ExactMatrix Q5 8 1 8}
    {w_391_3 : Fin 48 → ExactMatrix Q5 8 1 8}
    {w_391_4 : Fin 48 → ExactMatrix Q5 8 1 8}
    {w_391_5 : Fin 48 → ExactMatrix Q5 8 1 8}
    {w_392_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_392_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_392_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_392_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_392_4 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_392_5 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_4 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_5 : Fin 48 → ExactMatrix Q0 8 1 6}
    (h386_5 : (∀ i : Fin 240, Stage_online_eval_0_0.parallel_generatedRoot_386 Backend.backend model params i (w_371_0, (w_372_0, (w_373_0, (w_374_0, (w_375_0, (w_376_0, (w_377_0, (w_378_0, (w_379_0, (w_380_0, (w_381_0, (w_382_0, (w_383_0, (w_363_0, ((w_384_0 i), ((w_385_0 i), ())))))))))))))))) ((w_386_0 i), ((w_386_1 i), ((w_386_2 i), ((w_386_3 i), ((w_386_4 i), ((w_386_5 i), ()))))))))
    (h387_5 : (∀ i : Fin 48, Stage_online_eval_0_0.parallel_generatedRoot_387 Backend.backend model params i (w_386_0, (w_386_1, (w_386_2, (w_386_3, (w_386_4, (w_386_5, ())))))) ((w_387_0 i), ((w_387_1 i), ((w_387_2 i), ((w_387_3 i), ((w_387_4 i), ((w_387_5 i), ()))))))))
    (h390_5 : (∀ i : Fin 240, Stage_online_eval_0_0.parallel_generatedRoot_390 Backend.backend model params i (w_371_0, (w_372_0, (w_373_0, (w_374_0, (w_375_0, (w_376_0, (w_387_0, (w_387_2, (w_387_4, (w_380_0, (w_387_1, (w_387_3, (w_387_5, (w_363_0, ((w_388_0 i), ((w_389_0 i), ())))))))))))))))) ((w_390_0 i), ((w_390_1 i), ((w_390_2 i), ((w_390_3 i), ((w_390_4 i), ((w_390_5 i), ()))))))))
    (h391_5 : (∀ i : Fin 48, Stage_online_eval_0_0.parallel_generatedRoot_391 Backend.backend model params i (w_390_0, (w_390_1, (w_390_2, (w_390_3, (w_390_4, (w_390_5, ())))))) ((w_391_0 i), ((w_391_1 i), ((w_391_2 i), ((w_391_3 i), ((w_391_4 i), ((w_391_5 i), ()))))))))
    (h392_5 : (∀ i : Fin 48, Stage_online_eval_0_0.parallel_generatedRoot_392 Backend.backend model params i (w_365_0, (w_366_0, (w_367_0, (w_368_0, (w_369_0, (w_363_0, (w_370_0, ((w_391_4 i), ((w_391_0 i), ((w_391_2 i), ((w_391_5 i), ((w_391_1 i), ((w_391_3 i), ()))))))))))))) ((w_392_0 i), ((w_392_1 i), ((w_392_2 i), ((w_392_3 i), ((w_392_4 i), ((w_392_5 i), ()))))))))
    (h393_5 : (∀ i : Fin 48, Stage_online_eval_0_0.parallel_generatedRoot_393 Backend.backend model params i (w_358_0, (w_359_0, (w_360_0, (w_361_0, (w_362_0, (w_363_0, (w_364_0, ((w_392_4 i), ((w_392_0 i), ((w_392_2 i), ((w_392_5 i), ((w_392_1 i), ((w_392_3 i), ()))))))))))))) ((w_393_0 i), ((w_393_1 i), ((w_393_2 i), ((w_393_3 i), ((w_393_4 i), ((w_393_5 i), ()))))))))
    {pw_201_0 : ByteArray}
    {pw_207_0 : Fin 240 → Int}
    {pw_208_0 : Fin 240 → Int}
    {pw_212_0 : Fin 240 → Int}
    {pw_213_0 : Fin 240 → Int}
    {pw_202_0 : ExactMatrix Q7 8 1 12}
    {pw_203_0 : ExactMatrix Q7 8 1 12}
    {pw_204_0 : ExactMatrix Q7 8 1 12}
    {pw_205_0 : ExactMatrix Q7 8 1 12}
    {pw_206_0 : ExactMatrix Q7 8 1 1}
    {pw_209_0 : ExactMatrix Q7 8 1 1}
    {pw_210_0 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_210_1 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_210_2 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_210_3 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_210_4 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_210_5 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_210_6 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_210_7 : Fin 240 → ExactMatrix Q6 8 1 10}
    {pw_211_0 : Fin 48 → ExactMatrix Q6 8 1 10}
    {pw_211_1 : Fin 48 → ExactMatrix Q6 8 1 10}
    {pw_211_2 : Fin 48 → ExactMatrix Q6 8 1 10}
    {pw_211_3 : Fin 48 → ExactMatrix Q6 8 1 10}
    {pw_214_0 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_214_1 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_214_2 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_214_3 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_214_4 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_214_5 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_214_6 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_214_7 : Fin 240 → ExactMatrix Q5 8 1 8}
    {pw_215_0 : Fin 48 → ExactMatrix Q5 8 1 8}
    {pw_215_1 : Fin 48 → ExactMatrix Q5 8 1 8}
    {pw_215_2 : Fin 48 → ExactMatrix Q5 8 1 8}
    {pw_215_3 : Fin 48 → ExactMatrix Q5 8 1 8}
    {pw_216_0 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_216_1 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_216_2 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_216_3 : Fin 4 → ExactMatrix Q0 8 1 6}
    {pw_217_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_217_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_217_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_217_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_218_0 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_218_1 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_218_2 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_218_3 : Fin 4 → ExactMatrix Q1 8 1 6}
    {pw_219_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    (ph210_7 : (∀ i : Fin 240, Stage_preprocess_0_0.parallel_generatedRoot_210 Backend.backend model params' i (pw_201_0, (pw_202_0, (pw_203_0, (pw_204_0, (pw_205_0, (pw_206_0, ((pw_207_0 i), ((pw_208_0 i), (pw_209_0, ()))))))))) ((pw_210_0 i), ((pw_210_1 i), ((pw_210_2 i), ((pw_210_3 i), ((pw_210_4 i), ((pw_210_5 i), ((pw_210_6 i), ((pw_210_7 i), ()))))))))))
    (ph211_3 : (∀ i : Fin 48, Stage_preprocess_0_0.parallel_generatedRoot_211 Backend.backend model params' i (pw_210_0, (pw_210_1, (pw_210_2, (pw_210_3, ())))) ((pw_211_0 i), ((pw_211_1 i), ((pw_211_2 i), ((pw_211_3 i), ()))))))
    (ph214_7 : (∀ i : Fin 240, Stage_preprocess_0_0.parallel_generatedRoot_214 Backend.backend model params' i (pw_201_0, (pw_211_0, (pw_211_2, (pw_211_1, (pw_211_3, (pw_206_0, ((pw_212_0 i), ((pw_213_0 i), (pw_209_0, ()))))))))) ((pw_214_0 i), ((pw_214_1 i), ((pw_214_2 i), ((pw_214_3 i), ((pw_214_4 i), ((pw_214_5 i), ((pw_214_6 i), ((pw_214_7 i), ()))))))))))
    (ph215_3 : (∀ i : Fin 48, Stage_preprocess_0_0.parallel_generatedRoot_215 Backend.backend model params' i (pw_214_0, (pw_214_1, (pw_214_2, (pw_214_3, ())))) ((pw_215_0 i), ((pw_215_1 i), ((pw_215_2 i), ((pw_215_3 i), ()))))))
    (ph217_3 : (∀ i : Fin 48, Stage_preprocess_0_0.parallel_generatedRoot_217 Backend.backend model params' i (pw_201_0, ((pw_215_0 i), ((pw_215_2 i), (pw_216_0, ((pw_215_1 i), ((pw_215_3 i), (pw_216_2, (pw_216_1, (pw_216_3, ()))))))))) ((pw_217_0 i), ((pw_217_1 i), ((pw_217_2 i), ((pw_217_3 i), ()))))))
    (ph219_3 : (∀ i : Fin 48, Stage_preprocess_0_0.parallel_generatedRoot_219 Backend.backend model params' i (pw_201_0, ((pw_217_0 i), ((pw_217_2 i), (pw_218_0, ((pw_217_1 i), ((pw_217_3 i), (pw_218_2, (pw_218_1, (pw_218_3, ()))))))))) ((pw_219_0 i), ((pw_219_1 i), ((pw_219_2 i), ((pw_219_3 i), ()))))))
    (hk : w_363_0 = pw_201_0)
    (ho0 : w_377_0 = pw_202_0) (ho1 : w_378_0 = pw_203_0) (ho2 : w_381_0 = pw_204_0) (ho3 : w_382_0 = pw_205_0)
    (hf0 : w_365_0 = pw_216_0) (hf1 : w_366_0 = pw_216_1) (hf2 : w_367_0 = pw_216_2) (hf3 : w_368_0 = pw_216_3)
    (hg0 : w_358_0 = pw_218_0) (hg1 : w_359_0 = pw_218_1) (hg2 : w_360_0 = pw_218_2) (hg3 : w_361_0 = pw_218_3)
    (hc0 : w_384_0 = pw_210_6) (hc1 : w_385_0 = pw_210_7) (hc2 : w_388_0 = pw_214_6) (hc3 : w_389_0 = pw_214_7)
    (haA : ∀ (k : Fin 5) (c : Fin 4), ![w_372_0, w_375_0, w_373_0, w_376_0] c k =
      model.sample Q7 N 1 14 w_363_0 (completeHashTag labelPrefix
        [.bytes memberBytes, .u64Le (k : Int), .bytes componentBytes, .u64Le (c : Int)])) :
    ∀ o : Fin 48, w_393_0 o = pw_219_0 o ∧ w_393_1 o = pw_219_1 o ∧ w_393_2 o = pw_219_2 o ∧
      w_393_3 o = pw_219_3 o := by
  subst hk ho0 ho1 ho2 ho3 hf0 hf1 hf2 hf3 hg0 hg1 hg2 hg3 hc0 hc1 hc2 hc3
  have hc386 : ∀ i : Fin 240, w_386_0 i = pw_210_0 i ∧ w_386_1 i = pw_210_1 i ∧
      w_386_2 i = pw_210_2 i ∧ w_386_3 i = pw_210_3 i := fun i ↦
    keysCell386 (hon := h386_5 i) (hpre := ph210_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm386 : w_387_0 = pw_211_0 ∧ w_387_1 = pw_211_1 ∧ w_387_2 = pw_211_2 ∧ w_387_3 = pw_211_3 := by
    have a := fun o ↦ members387 (h387_5 o)
    have b := fun o ↦ premembers211 (ph211_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc386 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc386 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc386 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc386 _).2.2.2
  obtain ⟨m0, m1, m2, m3⟩ := hm386
  rw [m0, m1, m2, m3] at h390_5
  have hc390 : ∀ i : Fin 240, w_390_0 i = pw_214_0 i ∧ w_390_1 i = pw_214_1 i ∧
      w_390_2 i = pw_214_2 i ∧ w_390_3 i = pw_214_3 i := fun i ↦
    keysCell390 (hon := h390_5 i) (hpre := ph214_7 i) (haA := haA) (hbot0 := rfl) (hbot1 := rfl)
  have hm390 : w_391_0 = pw_215_0 ∧ w_391_1 = pw_215_1 ∧ w_391_2 = pw_215_2 ∧ w_391_3 = pw_215_3 := by
    have a := fun o ↦ members391 (h391_5 o)
    have b := fun o ↦ premembers215 (ph215_3 o)
    simp only [Prod.mk.injEq] at a b
    refine ⟨funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_, funext fun o ↦ ?_⟩
    · rw [(a o).1, (b o).1]; exact Finset.sum_congr rfl fun k _ ↦ (hc390 _).1
    · rw [(a o).2.1, (b o).2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc390 _).2.1
    · rw [(a o).2.2.1, (b o).2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc390 _).2.2.1
    · rw [(a o).2.2.2.1, (b o).2.2.2.1]; exact Finset.sum_congr rfl fun k _ ↦ (hc390 _).2.2.2
  obtain ⟨n0, n1, n2, n3⟩ := hm390
  rw [n0, n1, n2, n3] at h392_5
  have hr : ∀ o : Fin 48, w_392_0 o = pw_217_0 o ∧ w_392_1 o = pw_217_1 o ∧ w_392_2 o = pw_217_2 o ∧
      w_392_3 o = pw_217_3 o := by
    intro o
    have a := keyRound392 (h392_5 o)
    have b := preKeyRound217 (ph217_3 o)
    rw [← b] at a
    simp only [Prod.mk.injEq] at a
    exact ⟨a.1, a.2.1, a.2.2.1, a.2.2.2.1⟩
  intro o
  have a := keyRound393 (h393_5 o)
  have b := preKeyRound219 (ph219_3 o)
  obtain ⟨r0, r1, r2, r3⟩ := hr o
  rw [r0, r1, r2, r3, ← b] at a
  simp only [Prod.mk.injEq] at a
  exact ⟨a.1, a.2.1, a.2.2.1, a.2.2.2.1⟩

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Fresh column `0`: the online root's member sums of keys are the preprocessing's. -/
theorem freshKeyCol0 {model : HashModel} {params : Stage_online_eval_0_0.Params} {params' : Stage_preprocess_0_0.Params}
    {w_393_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_4 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_5 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_394_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_394_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_394_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_394_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_394_4 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_394_5 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_395_0 : ExactMatrix Q0 8 1 6}
    {w_396_0 : ExactMatrix Q0 8 1 6}
    {w_397_0 : ExactMatrix Q0 8 1 6}
    {w_398_0 : ExactMatrix Q0 8 1 6}
    {w_399_0 : ExactMatrix Q0 8 1 6}
    {w_400_0 : ExactMatrix Q0 8 1 6}
    {w_401_0 : ExactMatrix Q0 8 1 6}
    {w_402_0 : ExactMatrix Q0 8 1 6}
    {w_403_0 : ExactMatrix Q0 8 1 6}
    {w_404_0 : ExactMatrix Q0 8 1 6}
    {w_405_0 : ExactMatrix Q0 8 1 6}
    {w_406_0 : ExactMatrix Q0 8 1 6}
    {w_407_0 : ExactMatrix Q0 8 1 6}
    {w_408_0 : ExactMatrix Q0 8 1 6}
    {w_409_0 : ExactMatrix Q0 8 1 6}
    {w_410_0 : ExactMatrix Q0 8 1 6}
    {w_411_0 : ExactMatrix Q0 8 1 6}
    {w_412_0 : ExactMatrix Q0 8 1 6}
    {w_413_0 : ExactMatrix Q0 8 1 6}
    {w_413_1 : ExactMatrix Q0 8 1 6}
    {w_413_2 : ExactMatrix Q0 8 1 6}
    {w_413_3 : ExactMatrix Q0 8 1 6}
    {w_413_4 : ExactMatrix Q0 8 1 6}
    {w_413_5 : ExactMatrix Q0 8 1 6}
    (e_397_0 : w_397_0 = matrixAdd w_395_0 w_396_0)
    (e_400_0 : w_400_0 = matrixAdd w_398_0 w_399_0)
    (e_403_0 : w_403_0 = matrixAdd w_401_0 w_402_0)
    (e_406_0 : w_406_0 = matrixAdd w_404_0 w_405_0)
    (e_409_0 : w_409_0 = matrixAdd w_407_0 w_408_0)
    (e_412_0 : w_412_0 = matrixAdd w_410_0 w_411_0)
    (h394_5 : (∀ i : Fin 8, Stage_online_eval_0_0.parallel_generatedRoot_394 Backend.backend model params i ((w_393_0 ⟨i.val, by omega⟩), ((w_393_1 ⟨i.val, by omega⟩), ((w_393_2 ⟨i.val, by omega⟩), ((w_393_3 ⟨i.val, by omega⟩), ((w_393_4 ⟨i.val, by omega⟩), ((w_393_5 ⟨i.val, by omega⟩), ())))))) ((w_394_0 i), ((w_394_1 i), ((w_394_2 i), ((w_394_3 i), ((w_394_4 i), ((w_394_5 i), ()))))))))
    (h395_0 : familyGetStatic w_394_0 (0) w_395_0)
    (h396_0 : familyGetStatic w_394_0 (1) w_396_0)
    (h398_0 : familyGetStatic w_394_1 (0) w_398_0)
    (h399_0 : familyGetStatic w_394_1 (1) w_399_0)
    (h401_0 : familyGetStatic w_394_2 (0) w_401_0)
    (h402_0 : familyGetStatic w_394_2 (1) w_402_0)
    (h404_0 : familyGetStatic w_394_3 (0) w_404_0)
    (h405_0 : familyGetStatic w_394_3 (1) w_405_0)
    (h407_0 : familyGetStatic w_394_4 (0) w_407_0)
    (h408_0 : familyGetStatic w_394_4 (1) w_408_0)
    (h410_0 : familyGetStatic w_394_5 (0) w_410_0)
    (h411_0 : familyGetStatic w_394_5 (1) w_411_0)
    (h413_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_online_eval_0_0.sequential_generatedRoot_413 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_394_0, (w_394_1, (w_394_2, (w_394_3, (w_394_4, (w_394_5, ())))))))))))) next) (Int.toNat (6)) (w_397_0, (w_400_0, (w_403_0, (w_406_0, (w_409_0, (w_412_0, ())))))) (w_413_0, (w_413_1, (w_413_2, (w_413_3, (w_413_4, (w_413_5, ())))))))
    {pw_219_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_220_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_220_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_220_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_220_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_221_0 : ExactMatrix Q0 8 1 6}
    {pw_222_0 : ExactMatrix Q0 8 1 6}
    {pw_223_0 : ExactMatrix Q0 8 1 6}
    {pw_224_0 : ExactMatrix Q0 8 1 6}
    {pw_225_0 : ExactMatrix Q0 8 1 6}
    {pw_226_0 : ExactMatrix Q0 8 1 6}
    {pw_227_0 : ExactMatrix Q0 8 1 6}
    {pw_228_0 : ExactMatrix Q0 8 1 6}
    {pw_229_0 : ExactMatrix Q0 8 1 6}
    {pw_230_0 : ExactMatrix Q0 8 1 6}
    {pw_231_0 : ExactMatrix Q0 8 1 6}
    {pw_232_0 : ExactMatrix Q0 8 1 6}
    {pw_233_0 : ExactMatrix Q0 8 1 6}
    {pw_233_1 : ExactMatrix Q0 8 1 6}
    {pw_233_2 : ExactMatrix Q0 8 1 6}
    {pw_233_3 : ExactMatrix Q0 8 1 6}
    (pe_223_0 : pw_223_0 = matrixAdd pw_221_0 pw_222_0)
    (pe_226_0 : pw_226_0 = matrixAdd pw_224_0 pw_225_0)
    (pe_229_0 : pw_229_0 = matrixAdd pw_227_0 pw_228_0)
    (pe_232_0 : pw_232_0 = matrixAdd pw_230_0 pw_231_0)
    (ph220_3 : (∀ i : Fin 8, Stage_preprocess_0_0.parallel_generatedRoot_220 Backend.backend model params' i ((pw_219_0 ⟨i.val, by omega⟩), ((pw_219_1 ⟨i.val, by omega⟩), ((pw_219_2 ⟨i.val, by omega⟩), ((pw_219_3 ⟨i.val, by omega⟩), ())))) ((pw_220_0 i), ((pw_220_1 i), ((pw_220_2 i), ((pw_220_3 i), ()))))))
    (ph221_0 : familyGetStatic pw_220_0 (0) pw_221_0)
    (ph222_0 : familyGetStatic pw_220_0 (1) pw_222_0)
    (ph224_0 : familyGetStatic pw_220_1 (0) pw_224_0)
    (ph225_0 : familyGetStatic pw_220_1 (1) pw_225_0)
    (ph227_0 : familyGetStatic pw_220_2 (0) pw_227_0)
    (ph228_0 : familyGetStatic pw_220_2 (1) pw_228_0)
    (ph230_0 : familyGetStatic pw_220_3 (0) pw_230_0)
    (ph231_0 : familyGetStatic pw_220_3 (1) pw_231_0)
    (ph233_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_preprocess_0_0.sequential_generatedRoot_233 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_220_0, (pw_220_1, (pw_220_2, (pw_220_3, ())))))))) next) (Int.toNat (6)) (pw_223_0, (pw_226_0, (pw_229_0, (pw_232_0, ())))) (pw_233_0, (pw_233_1, (pw_233_2, (pw_233_3, ())))))
    (hs : ∀ o : Fin 48, w_393_0 o = pw_219_0 o ∧ w_393_1 o = pw_219_1 o ∧ w_393_2 o = pw_219_2 o ∧
      w_393_3 o = pw_219_3 o) :
    w_413_1 = pw_233_1 ∧ w_413_3 = pw_233_3 := by
  have hp : ∀ o : Fin 8, w_394_0 o = pw_220_0 o ∧ w_394_1 o = pw_220_1 o ∧ w_394_2 o = pw_220_2 o ∧
      w_394_3 o = pw_220_3 o := by
    intro o
    have a := place394 (h := h394_5 o)
      (x := (w_393_0 ⟨o.val, by omega⟩, w_393_1 ⟨o.val, by omega⟩,
        w_393_2 ⟨o.val, by omega⟩, w_393_3 ⟨o.val, by omega⟩,
        w_393_4 ⟨o.val, by omega⟩, w_393_5 ⟨o.val, by omega⟩, ()))
    have b := prePlace220 (h := ph220_3 o)
      (x := (pw_219_0 ⟨o.val, by omega⟩, pw_219_1 ⟨o.val, by omega⟩,
        pw_219_2 ⟨o.val, by omega⟩, pw_219_3 ⟨o.val, by omega⟩, ()))
    simp only [Prod.mk.injEq] at a b
    obtain ⟨s0, s1, s2, s3⟩ := hs ⟨o.val, by omega⟩
    exact ⟨by rw [a.1, b.1, s0], by rw [a.2.1, b.2.1, s1], by rw [a.2.2.1, b.2.2.1, s2],
      by rw [a.2.2.2.1, b.2.2.2.1, s3]⟩
  subst e_397_0 e_400_0 e_403_0 e_406_0 e_409_0 e_412_0 pe_223_0 pe_226_0 pe_229_0 pe_232_0
  have a := rootSum413 h395_0 h396_0 h398_0 h399_0 h401_0 h402_0 h404_0 h405_0 h407_0 h408_0 h410_0 h411_0 h413_5
  have b := prerootSum233 ph221_0 ph222_0 ph224_0 ph225_0 ph227_0 ph228_0 ph230_0 ph231_0 ph233_3
  simp only [Prod.mk.injEq] at a b
  rw [a.2.1, a.2.2.2.1, b.2.1, b.2.2.2.1]
  exact ⟨Finset.sum_congr rfl fun o _ ↦ (hp o).2.1, Finset.sum_congr rfl fun o _ ↦ (hp o).2.2.2⟩

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Fresh column `1`: the online root's member sums of keys are the preprocessing's. -/
theorem freshKeyCol1 {model : HashModel} {params : Stage_online_eval_0_0.Params} {params' : Stage_preprocess_0_0.Params}
    {w_393_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_4 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_5 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_419_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_419_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_419_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_419_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_419_4 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_419_5 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_420_0 : ExactMatrix Q0 8 1 6}
    {w_421_0 : ExactMatrix Q0 8 1 6}
    {w_422_0 : ExactMatrix Q0 8 1 6}
    {w_423_0 : ExactMatrix Q0 8 1 6}
    {w_424_0 : ExactMatrix Q0 8 1 6}
    {w_425_0 : ExactMatrix Q0 8 1 6}
    {w_426_0 : ExactMatrix Q0 8 1 6}
    {w_427_0 : ExactMatrix Q0 8 1 6}
    {w_428_0 : ExactMatrix Q0 8 1 6}
    {w_429_0 : ExactMatrix Q0 8 1 6}
    {w_430_0 : ExactMatrix Q0 8 1 6}
    {w_431_0 : ExactMatrix Q0 8 1 6}
    {w_432_0 : ExactMatrix Q0 8 1 6}
    {w_433_0 : ExactMatrix Q0 8 1 6}
    {w_434_0 : ExactMatrix Q0 8 1 6}
    {w_435_0 : ExactMatrix Q0 8 1 6}
    {w_436_0 : ExactMatrix Q0 8 1 6}
    {w_437_0 : ExactMatrix Q0 8 1 6}
    {w_438_0 : ExactMatrix Q0 8 1 6}
    {w_438_1 : ExactMatrix Q0 8 1 6}
    {w_438_2 : ExactMatrix Q0 8 1 6}
    {w_438_3 : ExactMatrix Q0 8 1 6}
    {w_438_4 : ExactMatrix Q0 8 1 6}
    {w_438_5 : ExactMatrix Q0 8 1 6}
    (e_422_0 : w_422_0 = matrixAdd w_420_0 w_421_0)
    (e_425_0 : w_425_0 = matrixAdd w_423_0 w_424_0)
    (e_428_0 : w_428_0 = matrixAdd w_426_0 w_427_0)
    (e_431_0 : w_431_0 = matrixAdd w_429_0 w_430_0)
    (e_434_0 : w_434_0 = matrixAdd w_432_0 w_433_0)
    (e_437_0 : w_437_0 = matrixAdd w_435_0 w_436_0)
    (h419_5 : (∀ i : Fin 8, Stage_online_eval_0_0.parallel_generatedRoot_419 Backend.backend model params i ((w_393_0 ⟨i.val + 8, by omega⟩), ((w_393_1 ⟨i.val + 8, by omega⟩), ((w_393_2 ⟨i.val + 8, by omega⟩), ((w_393_3 ⟨i.val + 8, by omega⟩), ((w_393_4 ⟨i.val + 8, by omega⟩), ((w_393_5 ⟨i.val + 8, by omega⟩), ())))))) ((w_419_0 i), ((w_419_1 i), ((w_419_2 i), ((w_419_3 i), ((w_419_4 i), ((w_419_5 i), ()))))))))
    (h420_0 : familyGetStatic w_419_0 (0) w_420_0)
    (h421_0 : familyGetStatic w_419_0 (1) w_421_0)
    (h423_0 : familyGetStatic w_419_1 (0) w_423_0)
    (h424_0 : familyGetStatic w_419_1 (1) w_424_0)
    (h426_0 : familyGetStatic w_419_2 (0) w_426_0)
    (h427_0 : familyGetStatic w_419_2 (1) w_427_0)
    (h429_0 : familyGetStatic w_419_3 (0) w_429_0)
    (h430_0 : familyGetStatic w_419_3 (1) w_430_0)
    (h432_0 : familyGetStatic w_419_4 (0) w_432_0)
    (h433_0 : familyGetStatic w_419_4 (1) w_433_0)
    (h435_0 : familyGetStatic w_419_5 (0) w_435_0)
    (h436_0 : familyGetStatic w_419_5 (1) w_436_0)
    (h438_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_online_eval_0_0.sequential_generatedRoot_438 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_419_0, (w_419_1, (w_419_2, (w_419_3, (w_419_4, (w_419_5, ())))))))))))) next) (Int.toNat (6)) (w_422_0, (w_425_0, (w_428_0, (w_431_0, (w_434_0, (w_437_0, ())))))) (w_438_0, (w_438_1, (w_438_2, (w_438_3, (w_438_4, (w_438_5, ())))))))
    {pw_219_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_239_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_239_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_239_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_239_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_240_0 : ExactMatrix Q0 8 1 6}
    {pw_241_0 : ExactMatrix Q0 8 1 6}
    {pw_242_0 : ExactMatrix Q0 8 1 6}
    {pw_243_0 : ExactMatrix Q0 8 1 6}
    {pw_244_0 : ExactMatrix Q0 8 1 6}
    {pw_245_0 : ExactMatrix Q0 8 1 6}
    {pw_246_0 : ExactMatrix Q0 8 1 6}
    {pw_247_0 : ExactMatrix Q0 8 1 6}
    {pw_248_0 : ExactMatrix Q0 8 1 6}
    {pw_249_0 : ExactMatrix Q0 8 1 6}
    {pw_250_0 : ExactMatrix Q0 8 1 6}
    {pw_251_0 : ExactMatrix Q0 8 1 6}
    {pw_252_0 : ExactMatrix Q0 8 1 6}
    {pw_252_1 : ExactMatrix Q0 8 1 6}
    {pw_252_2 : ExactMatrix Q0 8 1 6}
    {pw_252_3 : ExactMatrix Q0 8 1 6}
    (pe_242_0 : pw_242_0 = matrixAdd pw_240_0 pw_241_0)
    (pe_245_0 : pw_245_0 = matrixAdd pw_243_0 pw_244_0)
    (pe_248_0 : pw_248_0 = matrixAdd pw_246_0 pw_247_0)
    (pe_251_0 : pw_251_0 = matrixAdd pw_249_0 pw_250_0)
    (ph239_3 : (∀ i : Fin 8, Stage_preprocess_0_0.parallel_generatedRoot_239 Backend.backend model params' i ((pw_219_0 ⟨i.val + 8, by omega⟩), ((pw_219_1 ⟨i.val + 8, by omega⟩), ((pw_219_2 ⟨i.val + 8, by omega⟩), ((pw_219_3 ⟨i.val + 8, by omega⟩), ())))) ((pw_239_0 i), ((pw_239_1 i), ((pw_239_2 i), ((pw_239_3 i), ()))))))
    (ph240_0 : familyGetStatic pw_239_0 (0) pw_240_0)
    (ph241_0 : familyGetStatic pw_239_0 (1) pw_241_0)
    (ph243_0 : familyGetStatic pw_239_1 (0) pw_243_0)
    (ph244_0 : familyGetStatic pw_239_1 (1) pw_244_0)
    (ph246_0 : familyGetStatic pw_239_2 (0) pw_246_0)
    (ph247_0 : familyGetStatic pw_239_2 (1) pw_247_0)
    (ph249_0 : familyGetStatic pw_239_3 (0) pw_249_0)
    (ph250_0 : familyGetStatic pw_239_3 (1) pw_250_0)
    (ph252_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_preprocess_0_0.sequential_generatedRoot_252 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_239_0, (pw_239_1, (pw_239_2, (pw_239_3, ())))))))) next) (Int.toNat (6)) (pw_242_0, (pw_245_0, (pw_248_0, (pw_251_0, ())))) (pw_252_0, (pw_252_1, (pw_252_2, (pw_252_3, ())))))
    (hs : ∀ o : Fin 48, w_393_0 o = pw_219_0 o ∧ w_393_1 o = pw_219_1 o ∧ w_393_2 o = pw_219_2 o ∧
      w_393_3 o = pw_219_3 o) :
    w_438_1 = pw_252_1 ∧ w_438_3 = pw_252_3 := by
  have hp : ∀ o : Fin 8, w_419_0 o = pw_239_0 o ∧ w_419_1 o = pw_239_1 o ∧ w_419_2 o = pw_239_2 o ∧
      w_419_3 o = pw_239_3 o := by
    intro o
    have a := place419 (h := h419_5 o)
      (x := (w_393_0 ⟨o.val + 8, by omega⟩, w_393_1 ⟨o.val + 8, by omega⟩,
        w_393_2 ⟨o.val + 8, by omega⟩, w_393_3 ⟨o.val + 8, by omega⟩,
        w_393_4 ⟨o.val + 8, by omega⟩, w_393_5 ⟨o.val + 8, by omega⟩, ()))
    have b := prePlace239 (h := ph239_3 o)
      (x := (pw_219_0 ⟨o.val + 8, by omega⟩, pw_219_1 ⟨o.val + 8, by omega⟩,
        pw_219_2 ⟨o.val + 8, by omega⟩, pw_219_3 ⟨o.val + 8, by omega⟩, ()))
    simp only [Prod.mk.injEq] at a b
    obtain ⟨s0, s1, s2, s3⟩ := hs ⟨o.val + 8, by omega⟩
    exact ⟨by rw [a.1, b.1, s0], by rw [a.2.1, b.2.1, s1], by rw [a.2.2.1, b.2.2.1, s2],
      by rw [a.2.2.2.1, b.2.2.2.1, s3]⟩
  subst e_422_0 e_425_0 e_428_0 e_431_0 e_434_0 e_437_0 pe_242_0 pe_245_0 pe_248_0 pe_251_0
  have a := rootSum438 h420_0 h421_0 h423_0 h424_0 h426_0 h427_0 h429_0 h430_0 h432_0 h433_0 h435_0 h436_0 h438_5
  have b := prerootSum252 ph240_0 ph241_0 ph243_0 ph244_0 ph246_0 ph247_0 ph249_0 ph250_0 ph252_3
  simp only [Prod.mk.injEq] at a b
  rw [a.2.1, a.2.2.2.1, b.2.1, b.2.2.2.1]
  exact ⟨Finset.sum_congr rfl fun o _ ↦ (hp o).2.1, Finset.sum_congr rfl fun o _ ↦ (hp o).2.2.2⟩

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Fresh column `2`: the online root's member sums of keys are the preprocessing's. -/
theorem freshKeyCol2 {model : HashModel} {params : Stage_online_eval_0_0.Params} {params' : Stage_preprocess_0_0.Params}
    {w_393_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_4 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_5 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_441_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_441_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_441_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_441_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_441_4 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_441_5 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_442_0 : ExactMatrix Q0 8 1 6}
    {w_443_0 : ExactMatrix Q0 8 1 6}
    {w_444_0 : ExactMatrix Q0 8 1 6}
    {w_445_0 : ExactMatrix Q0 8 1 6}
    {w_446_0 : ExactMatrix Q0 8 1 6}
    {w_447_0 : ExactMatrix Q0 8 1 6}
    {w_448_0 : ExactMatrix Q0 8 1 6}
    {w_449_0 : ExactMatrix Q0 8 1 6}
    {w_450_0 : ExactMatrix Q0 8 1 6}
    {w_451_0 : ExactMatrix Q0 8 1 6}
    {w_452_0 : ExactMatrix Q0 8 1 6}
    {w_453_0 : ExactMatrix Q0 8 1 6}
    {w_454_0 : ExactMatrix Q0 8 1 6}
    {w_455_0 : ExactMatrix Q0 8 1 6}
    {w_456_0 : ExactMatrix Q0 8 1 6}
    {w_457_0 : ExactMatrix Q0 8 1 6}
    {w_458_0 : ExactMatrix Q0 8 1 6}
    {w_459_0 : ExactMatrix Q0 8 1 6}
    {w_460_0 : ExactMatrix Q0 8 1 6}
    {w_460_1 : ExactMatrix Q0 8 1 6}
    {w_460_2 : ExactMatrix Q0 8 1 6}
    {w_460_3 : ExactMatrix Q0 8 1 6}
    {w_460_4 : ExactMatrix Q0 8 1 6}
    {w_460_5 : ExactMatrix Q0 8 1 6}
    (e_444_0 : w_444_0 = matrixAdd w_442_0 w_443_0)
    (e_447_0 : w_447_0 = matrixAdd w_445_0 w_446_0)
    (e_450_0 : w_450_0 = matrixAdd w_448_0 w_449_0)
    (e_453_0 : w_453_0 = matrixAdd w_451_0 w_452_0)
    (e_456_0 : w_456_0 = matrixAdd w_454_0 w_455_0)
    (e_459_0 : w_459_0 = matrixAdd w_457_0 w_458_0)
    (h441_5 : (∀ i : Fin 8, Stage_online_eval_0_0.parallel_generatedRoot_441 Backend.backend model params i ((w_393_0 ⟨i.val + 16, by omega⟩), ((w_393_1 ⟨i.val + 16, by omega⟩), ((w_393_2 ⟨i.val + 16, by omega⟩), ((w_393_3 ⟨i.val + 16, by omega⟩), ((w_393_4 ⟨i.val + 16, by omega⟩), ((w_393_5 ⟨i.val + 16, by omega⟩), ())))))) ((w_441_0 i), ((w_441_1 i), ((w_441_2 i), ((w_441_3 i), ((w_441_4 i), ((w_441_5 i), ()))))))))
    (h442_0 : familyGetStatic w_441_0 (0) w_442_0)
    (h443_0 : familyGetStatic w_441_0 (1) w_443_0)
    (h445_0 : familyGetStatic w_441_1 (0) w_445_0)
    (h446_0 : familyGetStatic w_441_1 (1) w_446_0)
    (h448_0 : familyGetStatic w_441_2 (0) w_448_0)
    (h449_0 : familyGetStatic w_441_2 (1) w_449_0)
    (h451_0 : familyGetStatic w_441_3 (0) w_451_0)
    (h452_0 : familyGetStatic w_441_3 (1) w_452_0)
    (h454_0 : familyGetStatic w_441_4 (0) w_454_0)
    (h455_0 : familyGetStatic w_441_4 (1) w_455_0)
    (h457_0 : familyGetStatic w_441_5 (0) w_457_0)
    (h458_0 : familyGetStatic w_441_5 (1) w_458_0)
    (h460_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_online_eval_0_0.sequential_generatedRoot_460 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_441_0, (w_441_1, (w_441_2, (w_441_3, (w_441_4, (w_441_5, ())))))))))))) next) (Int.toNat (6)) (w_444_0, (w_447_0, (w_450_0, (w_453_0, (w_456_0, (w_459_0, ())))))) (w_460_0, (w_460_1, (w_460_2, (w_460_3, (w_460_4, (w_460_5, ())))))))
    {pw_219_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_255_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_255_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_255_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_255_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_256_0 : ExactMatrix Q0 8 1 6}
    {pw_257_0 : ExactMatrix Q0 8 1 6}
    {pw_258_0 : ExactMatrix Q0 8 1 6}
    {pw_259_0 : ExactMatrix Q0 8 1 6}
    {pw_260_0 : ExactMatrix Q0 8 1 6}
    {pw_261_0 : ExactMatrix Q0 8 1 6}
    {pw_262_0 : ExactMatrix Q0 8 1 6}
    {pw_263_0 : ExactMatrix Q0 8 1 6}
    {pw_264_0 : ExactMatrix Q0 8 1 6}
    {pw_265_0 : ExactMatrix Q0 8 1 6}
    {pw_266_0 : ExactMatrix Q0 8 1 6}
    {pw_267_0 : ExactMatrix Q0 8 1 6}
    {pw_268_0 : ExactMatrix Q0 8 1 6}
    {pw_268_1 : ExactMatrix Q0 8 1 6}
    {pw_268_2 : ExactMatrix Q0 8 1 6}
    {pw_268_3 : ExactMatrix Q0 8 1 6}
    (pe_258_0 : pw_258_0 = matrixAdd pw_256_0 pw_257_0)
    (pe_261_0 : pw_261_0 = matrixAdd pw_259_0 pw_260_0)
    (pe_264_0 : pw_264_0 = matrixAdd pw_262_0 pw_263_0)
    (pe_267_0 : pw_267_0 = matrixAdd pw_265_0 pw_266_0)
    (ph255_3 : (∀ i : Fin 8, Stage_preprocess_0_0.parallel_generatedRoot_255 Backend.backend model params' i ((pw_219_0 ⟨i.val + 16, by omega⟩), ((pw_219_1 ⟨i.val + 16, by omega⟩), ((pw_219_2 ⟨i.val + 16, by omega⟩), ((pw_219_3 ⟨i.val + 16, by omega⟩), ())))) ((pw_255_0 i), ((pw_255_1 i), ((pw_255_2 i), ((pw_255_3 i), ()))))))
    (ph256_0 : familyGetStatic pw_255_0 (0) pw_256_0)
    (ph257_0 : familyGetStatic pw_255_0 (1) pw_257_0)
    (ph259_0 : familyGetStatic pw_255_1 (0) pw_259_0)
    (ph260_0 : familyGetStatic pw_255_1 (1) pw_260_0)
    (ph262_0 : familyGetStatic pw_255_2 (0) pw_262_0)
    (ph263_0 : familyGetStatic pw_255_2 (1) pw_263_0)
    (ph265_0 : familyGetStatic pw_255_3 (0) pw_265_0)
    (ph266_0 : familyGetStatic pw_255_3 (1) pw_266_0)
    (ph268_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_preprocess_0_0.sequential_generatedRoot_268 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_255_0, (pw_255_1, (pw_255_2, (pw_255_3, ())))))))) next) (Int.toNat (6)) (pw_258_0, (pw_261_0, (pw_264_0, (pw_267_0, ())))) (pw_268_0, (pw_268_1, (pw_268_2, (pw_268_3, ())))))
    (hs : ∀ o : Fin 48, w_393_0 o = pw_219_0 o ∧ w_393_1 o = pw_219_1 o ∧ w_393_2 o = pw_219_2 o ∧
      w_393_3 o = pw_219_3 o) :
    w_460_1 = pw_268_1 ∧ w_460_3 = pw_268_3 := by
  have hp : ∀ o : Fin 8, w_441_0 o = pw_255_0 o ∧ w_441_1 o = pw_255_1 o ∧ w_441_2 o = pw_255_2 o ∧
      w_441_3 o = pw_255_3 o := by
    intro o
    have a := place441 (h := h441_5 o)
      (x := (w_393_0 ⟨o.val + 16, by omega⟩, w_393_1 ⟨o.val + 16, by omega⟩,
        w_393_2 ⟨o.val + 16, by omega⟩, w_393_3 ⟨o.val + 16, by omega⟩,
        w_393_4 ⟨o.val + 16, by omega⟩, w_393_5 ⟨o.val + 16, by omega⟩, ()))
    have b := prePlace255 (h := ph255_3 o)
      (x := (pw_219_0 ⟨o.val + 16, by omega⟩, pw_219_1 ⟨o.val + 16, by omega⟩,
        pw_219_2 ⟨o.val + 16, by omega⟩, pw_219_3 ⟨o.val + 16, by omega⟩, ()))
    simp only [Prod.mk.injEq] at a b
    obtain ⟨s0, s1, s2, s3⟩ := hs ⟨o.val + 16, by omega⟩
    exact ⟨by rw [a.1, b.1, s0], by rw [a.2.1, b.2.1, s1], by rw [a.2.2.1, b.2.2.1, s2],
      by rw [a.2.2.2.1, b.2.2.2.1, s3]⟩
  subst e_444_0 e_447_0 e_450_0 e_453_0 e_456_0 e_459_0 pe_258_0 pe_261_0 pe_264_0 pe_267_0
  have a := rootSum460 h442_0 h443_0 h445_0 h446_0 h448_0 h449_0 h451_0 h452_0 h454_0 h455_0 h457_0 h458_0 h460_5
  have b := prerootSum268 ph256_0 ph257_0 ph259_0 ph260_0 ph262_0 ph263_0 ph265_0 ph266_0 ph268_3
  simp only [Prod.mk.injEq] at a b
  rw [a.2.1, a.2.2.2.1, b.2.1, b.2.2.2.1]
  exact ⟨Finset.sum_congr rfl fun o _ ↦ (hp o).2.1, Finset.sum_congr rfl fun o _ ↦ (hp o).2.2.2⟩

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Fresh column `3`: the online root's member sums of keys are the preprocessing's. -/
theorem freshKeyCol3 {model : HashModel} {params : Stage_online_eval_0_0.Params} {params' : Stage_preprocess_0_0.Params}
    {w_393_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_4 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_5 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_463_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_463_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_463_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_463_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_463_4 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_463_5 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_464_0 : ExactMatrix Q0 8 1 6}
    {w_465_0 : ExactMatrix Q0 8 1 6}
    {w_466_0 : ExactMatrix Q0 8 1 6}
    {w_467_0 : ExactMatrix Q0 8 1 6}
    {w_468_0 : ExactMatrix Q0 8 1 6}
    {w_469_0 : ExactMatrix Q0 8 1 6}
    {w_470_0 : ExactMatrix Q0 8 1 6}
    {w_471_0 : ExactMatrix Q0 8 1 6}
    {w_472_0 : ExactMatrix Q0 8 1 6}
    {w_473_0 : ExactMatrix Q0 8 1 6}
    {w_474_0 : ExactMatrix Q0 8 1 6}
    {w_475_0 : ExactMatrix Q0 8 1 6}
    {w_476_0 : ExactMatrix Q0 8 1 6}
    {w_477_0 : ExactMatrix Q0 8 1 6}
    {w_478_0 : ExactMatrix Q0 8 1 6}
    {w_479_0 : ExactMatrix Q0 8 1 6}
    {w_480_0 : ExactMatrix Q0 8 1 6}
    {w_481_0 : ExactMatrix Q0 8 1 6}
    {w_482_0 : ExactMatrix Q0 8 1 6}
    {w_482_1 : ExactMatrix Q0 8 1 6}
    {w_482_2 : ExactMatrix Q0 8 1 6}
    {w_482_3 : ExactMatrix Q0 8 1 6}
    {w_482_4 : ExactMatrix Q0 8 1 6}
    {w_482_5 : ExactMatrix Q0 8 1 6}
    (e_466_0 : w_466_0 = matrixAdd w_464_0 w_465_0)
    (e_469_0 : w_469_0 = matrixAdd w_467_0 w_468_0)
    (e_472_0 : w_472_0 = matrixAdd w_470_0 w_471_0)
    (e_475_0 : w_475_0 = matrixAdd w_473_0 w_474_0)
    (e_478_0 : w_478_0 = matrixAdd w_476_0 w_477_0)
    (e_481_0 : w_481_0 = matrixAdd w_479_0 w_480_0)
    (h463_5 : (∀ i : Fin 8, Stage_online_eval_0_0.parallel_generatedRoot_463 Backend.backend model params i ((w_393_0 ⟨i.val + 24, by omega⟩), ((w_393_1 ⟨i.val + 24, by omega⟩), ((w_393_2 ⟨i.val + 24, by omega⟩), ((w_393_3 ⟨i.val + 24, by omega⟩), ((w_393_4 ⟨i.val + 24, by omega⟩), ((w_393_5 ⟨i.val + 24, by omega⟩), ())))))) ((w_463_0 i), ((w_463_1 i), ((w_463_2 i), ((w_463_3 i), ((w_463_4 i), ((w_463_5 i), ()))))))))
    (h464_0 : familyGetStatic w_463_0 (0) w_464_0)
    (h465_0 : familyGetStatic w_463_0 (1) w_465_0)
    (h467_0 : familyGetStatic w_463_1 (0) w_467_0)
    (h468_0 : familyGetStatic w_463_1 (1) w_468_0)
    (h470_0 : familyGetStatic w_463_2 (0) w_470_0)
    (h471_0 : familyGetStatic w_463_2 (1) w_471_0)
    (h473_0 : familyGetStatic w_463_3 (0) w_473_0)
    (h474_0 : familyGetStatic w_463_3 (1) w_474_0)
    (h476_0 : familyGetStatic w_463_4 (0) w_476_0)
    (h477_0 : familyGetStatic w_463_4 (1) w_477_0)
    (h479_0 : familyGetStatic w_463_5 (0) w_479_0)
    (h480_0 : familyGetStatic w_463_5 (1) w_480_0)
    (h482_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_online_eval_0_0.sequential_generatedRoot_482 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_463_0, (w_463_1, (w_463_2, (w_463_3, (w_463_4, (w_463_5, ())))))))))))) next) (Int.toNat (6)) (w_466_0, (w_469_0, (w_472_0, (w_475_0, (w_478_0, (w_481_0, ())))))) (w_482_0, (w_482_1, (w_482_2, (w_482_3, (w_482_4, (w_482_5, ())))))))
    {pw_219_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_271_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_271_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_271_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_271_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_272_0 : ExactMatrix Q0 8 1 6}
    {pw_273_0 : ExactMatrix Q0 8 1 6}
    {pw_274_0 : ExactMatrix Q0 8 1 6}
    {pw_275_0 : ExactMatrix Q0 8 1 6}
    {pw_276_0 : ExactMatrix Q0 8 1 6}
    {pw_277_0 : ExactMatrix Q0 8 1 6}
    {pw_278_0 : ExactMatrix Q0 8 1 6}
    {pw_279_0 : ExactMatrix Q0 8 1 6}
    {pw_280_0 : ExactMatrix Q0 8 1 6}
    {pw_281_0 : ExactMatrix Q0 8 1 6}
    {pw_282_0 : ExactMatrix Q0 8 1 6}
    {pw_283_0 : ExactMatrix Q0 8 1 6}
    {pw_284_0 : ExactMatrix Q0 8 1 6}
    {pw_284_1 : ExactMatrix Q0 8 1 6}
    {pw_284_2 : ExactMatrix Q0 8 1 6}
    {pw_284_3 : ExactMatrix Q0 8 1 6}
    (pe_274_0 : pw_274_0 = matrixAdd pw_272_0 pw_273_0)
    (pe_277_0 : pw_277_0 = matrixAdd pw_275_0 pw_276_0)
    (pe_280_0 : pw_280_0 = matrixAdd pw_278_0 pw_279_0)
    (pe_283_0 : pw_283_0 = matrixAdd pw_281_0 pw_282_0)
    (ph271_3 : (∀ i : Fin 8, Stage_preprocess_0_0.parallel_generatedRoot_271 Backend.backend model params' i ((pw_219_0 ⟨i.val + 24, by omega⟩), ((pw_219_1 ⟨i.val + 24, by omega⟩), ((pw_219_2 ⟨i.val + 24, by omega⟩), ((pw_219_3 ⟨i.val + 24, by omega⟩), ())))) ((pw_271_0 i), ((pw_271_1 i), ((pw_271_2 i), ((pw_271_3 i), ()))))))
    (ph272_0 : familyGetStatic pw_271_0 (0) pw_272_0)
    (ph273_0 : familyGetStatic pw_271_0 (1) pw_273_0)
    (ph275_0 : familyGetStatic pw_271_1 (0) pw_275_0)
    (ph276_0 : familyGetStatic pw_271_1 (1) pw_276_0)
    (ph278_0 : familyGetStatic pw_271_2 (0) pw_278_0)
    (ph279_0 : familyGetStatic pw_271_2 (1) pw_279_0)
    (ph281_0 : familyGetStatic pw_271_3 (0) pw_281_0)
    (ph282_0 : familyGetStatic pw_271_3 (1) pw_282_0)
    (ph284_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_preprocess_0_0.sequential_generatedRoot_284 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_271_0, (pw_271_1, (pw_271_2, (pw_271_3, ())))))))) next) (Int.toNat (6)) (pw_274_0, (pw_277_0, (pw_280_0, (pw_283_0, ())))) (pw_284_0, (pw_284_1, (pw_284_2, (pw_284_3, ())))))
    (hs : ∀ o : Fin 48, w_393_0 o = pw_219_0 o ∧ w_393_1 o = pw_219_1 o ∧ w_393_2 o = pw_219_2 o ∧
      w_393_3 o = pw_219_3 o) :
    w_482_1 = pw_284_1 ∧ w_482_3 = pw_284_3 := by
  have hp : ∀ o : Fin 8, w_463_0 o = pw_271_0 o ∧ w_463_1 o = pw_271_1 o ∧ w_463_2 o = pw_271_2 o ∧
      w_463_3 o = pw_271_3 o := by
    intro o
    have a := place463 (h := h463_5 o)
      (x := (w_393_0 ⟨o.val + 24, by omega⟩, w_393_1 ⟨o.val + 24, by omega⟩,
        w_393_2 ⟨o.val + 24, by omega⟩, w_393_3 ⟨o.val + 24, by omega⟩,
        w_393_4 ⟨o.val + 24, by omega⟩, w_393_5 ⟨o.val + 24, by omega⟩, ()))
    have b := prePlace271 (h := ph271_3 o)
      (x := (pw_219_0 ⟨o.val + 24, by omega⟩, pw_219_1 ⟨o.val + 24, by omega⟩,
        pw_219_2 ⟨o.val + 24, by omega⟩, pw_219_3 ⟨o.val + 24, by omega⟩, ()))
    simp only [Prod.mk.injEq] at a b
    obtain ⟨s0, s1, s2, s3⟩ := hs ⟨o.val + 24, by omega⟩
    exact ⟨by rw [a.1, b.1, s0], by rw [a.2.1, b.2.1, s1], by rw [a.2.2.1, b.2.2.1, s2],
      by rw [a.2.2.2.1, b.2.2.2.1, s3]⟩
  subst e_466_0 e_469_0 e_472_0 e_475_0 e_478_0 e_481_0 pe_274_0 pe_277_0 pe_280_0 pe_283_0
  have a := rootSum482 h464_0 h465_0 h467_0 h468_0 h470_0 h471_0 h473_0 h474_0 h476_0 h477_0 h479_0 h480_0 h482_5
  have b := prerootSum284 ph272_0 ph273_0 ph275_0 ph276_0 ph278_0 ph279_0 ph281_0 ph282_0 ph284_3
  simp only [Prod.mk.injEq] at a b
  rw [a.2.1, a.2.2.2.1, b.2.1, b.2.2.2.1]
  exact ⟨Finset.sum_congr rfl fun o _ ↦ (hp o).2.1, Finset.sum_congr rfl fun o _ ↦ (hp o).2.2.2⟩

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Fresh column `4`: the online root's member sums of keys are the preprocessing's. -/
theorem freshKeyCol4 {model : HashModel} {params : Stage_online_eval_0_0.Params} {params' : Stage_preprocess_0_0.Params}
    {w_393_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_4 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_5 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_485_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_485_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_485_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_485_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_485_4 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_485_5 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_486_0 : ExactMatrix Q0 8 1 6}
    {w_487_0 : ExactMatrix Q0 8 1 6}
    {w_488_0 : ExactMatrix Q0 8 1 6}
    {w_489_0 : ExactMatrix Q0 8 1 6}
    {w_490_0 : ExactMatrix Q0 8 1 6}
    {w_491_0 : ExactMatrix Q0 8 1 6}
    {w_492_0 : ExactMatrix Q0 8 1 6}
    {w_493_0 : ExactMatrix Q0 8 1 6}
    {w_494_0 : ExactMatrix Q0 8 1 6}
    {w_495_0 : ExactMatrix Q0 8 1 6}
    {w_496_0 : ExactMatrix Q0 8 1 6}
    {w_497_0 : ExactMatrix Q0 8 1 6}
    {w_498_0 : ExactMatrix Q0 8 1 6}
    {w_499_0 : ExactMatrix Q0 8 1 6}
    {w_500_0 : ExactMatrix Q0 8 1 6}
    {w_501_0 : ExactMatrix Q0 8 1 6}
    {w_502_0 : ExactMatrix Q0 8 1 6}
    {w_503_0 : ExactMatrix Q0 8 1 6}
    {w_504_0 : ExactMatrix Q0 8 1 6}
    {w_504_1 : ExactMatrix Q0 8 1 6}
    {w_504_2 : ExactMatrix Q0 8 1 6}
    {w_504_3 : ExactMatrix Q0 8 1 6}
    {w_504_4 : ExactMatrix Q0 8 1 6}
    {w_504_5 : ExactMatrix Q0 8 1 6}
    (e_488_0 : w_488_0 = matrixAdd w_486_0 w_487_0)
    (e_491_0 : w_491_0 = matrixAdd w_489_0 w_490_0)
    (e_494_0 : w_494_0 = matrixAdd w_492_0 w_493_0)
    (e_497_0 : w_497_0 = matrixAdd w_495_0 w_496_0)
    (e_500_0 : w_500_0 = matrixAdd w_498_0 w_499_0)
    (e_503_0 : w_503_0 = matrixAdd w_501_0 w_502_0)
    (h485_5 : (∀ i : Fin 8, Stage_online_eval_0_0.parallel_generatedRoot_485 Backend.backend model params i ((w_393_0 ⟨i.val + 32, by omega⟩), ((w_393_1 ⟨i.val + 32, by omega⟩), ((w_393_2 ⟨i.val + 32, by omega⟩), ((w_393_3 ⟨i.val + 32, by omega⟩), ((w_393_4 ⟨i.val + 32, by omega⟩), ((w_393_5 ⟨i.val + 32, by omega⟩), ())))))) ((w_485_0 i), ((w_485_1 i), ((w_485_2 i), ((w_485_3 i), ((w_485_4 i), ((w_485_5 i), ()))))))))
    (h486_0 : familyGetStatic w_485_0 (0) w_486_0)
    (h487_0 : familyGetStatic w_485_0 (1) w_487_0)
    (h489_0 : familyGetStatic w_485_1 (0) w_489_0)
    (h490_0 : familyGetStatic w_485_1 (1) w_490_0)
    (h492_0 : familyGetStatic w_485_2 (0) w_492_0)
    (h493_0 : familyGetStatic w_485_2 (1) w_493_0)
    (h495_0 : familyGetStatic w_485_3 (0) w_495_0)
    (h496_0 : familyGetStatic w_485_3 (1) w_496_0)
    (h498_0 : familyGetStatic w_485_4 (0) w_498_0)
    (h499_0 : familyGetStatic w_485_4 (1) w_499_0)
    (h501_0 : familyGetStatic w_485_5 (0) w_501_0)
    (h502_0 : familyGetStatic w_485_5 (1) w_502_0)
    (h504_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_online_eval_0_0.sequential_generatedRoot_504 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_485_0, (w_485_1, (w_485_2, (w_485_3, (w_485_4, (w_485_5, ())))))))))))) next) (Int.toNat (6)) (w_488_0, (w_491_0, (w_494_0, (w_497_0, (w_500_0, (w_503_0, ())))))) (w_504_0, (w_504_1, (w_504_2, (w_504_3, (w_504_4, (w_504_5, ())))))))
    {pw_219_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_287_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_287_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_287_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_287_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_288_0 : ExactMatrix Q0 8 1 6}
    {pw_289_0 : ExactMatrix Q0 8 1 6}
    {pw_290_0 : ExactMatrix Q0 8 1 6}
    {pw_291_0 : ExactMatrix Q0 8 1 6}
    {pw_292_0 : ExactMatrix Q0 8 1 6}
    {pw_293_0 : ExactMatrix Q0 8 1 6}
    {pw_294_0 : ExactMatrix Q0 8 1 6}
    {pw_295_0 : ExactMatrix Q0 8 1 6}
    {pw_296_0 : ExactMatrix Q0 8 1 6}
    {pw_297_0 : ExactMatrix Q0 8 1 6}
    {pw_298_0 : ExactMatrix Q0 8 1 6}
    {pw_299_0 : ExactMatrix Q0 8 1 6}
    {pw_300_0 : ExactMatrix Q0 8 1 6}
    {pw_300_1 : ExactMatrix Q0 8 1 6}
    {pw_300_2 : ExactMatrix Q0 8 1 6}
    {pw_300_3 : ExactMatrix Q0 8 1 6}
    (pe_290_0 : pw_290_0 = matrixAdd pw_288_0 pw_289_0)
    (pe_293_0 : pw_293_0 = matrixAdd pw_291_0 pw_292_0)
    (pe_296_0 : pw_296_0 = matrixAdd pw_294_0 pw_295_0)
    (pe_299_0 : pw_299_0 = matrixAdd pw_297_0 pw_298_0)
    (ph287_3 : (∀ i : Fin 8, Stage_preprocess_0_0.parallel_generatedRoot_287 Backend.backend model params' i ((pw_219_0 ⟨i.val + 32, by omega⟩), ((pw_219_1 ⟨i.val + 32, by omega⟩), ((pw_219_2 ⟨i.val + 32, by omega⟩), ((pw_219_3 ⟨i.val + 32, by omega⟩), ())))) ((pw_287_0 i), ((pw_287_1 i), ((pw_287_2 i), ((pw_287_3 i), ()))))))
    (ph288_0 : familyGetStatic pw_287_0 (0) pw_288_0)
    (ph289_0 : familyGetStatic pw_287_0 (1) pw_289_0)
    (ph291_0 : familyGetStatic pw_287_1 (0) pw_291_0)
    (ph292_0 : familyGetStatic pw_287_1 (1) pw_292_0)
    (ph294_0 : familyGetStatic pw_287_2 (0) pw_294_0)
    (ph295_0 : familyGetStatic pw_287_2 (1) pw_295_0)
    (ph297_0 : familyGetStatic pw_287_3 (0) pw_297_0)
    (ph298_0 : familyGetStatic pw_287_3 (1) pw_298_0)
    (ph300_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_preprocess_0_0.sequential_generatedRoot_300 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_287_0, (pw_287_1, (pw_287_2, (pw_287_3, ())))))))) next) (Int.toNat (6)) (pw_290_0, (pw_293_0, (pw_296_0, (pw_299_0, ())))) (pw_300_0, (pw_300_1, (pw_300_2, (pw_300_3, ())))))
    (hs : ∀ o : Fin 48, w_393_0 o = pw_219_0 o ∧ w_393_1 o = pw_219_1 o ∧ w_393_2 o = pw_219_2 o ∧
      w_393_3 o = pw_219_3 o) :
    w_504_1 = pw_300_1 ∧ w_504_3 = pw_300_3 := by
  have hp : ∀ o : Fin 8, w_485_0 o = pw_287_0 o ∧ w_485_1 o = pw_287_1 o ∧ w_485_2 o = pw_287_2 o ∧
      w_485_3 o = pw_287_3 o := by
    intro o
    have a := place485 (h := h485_5 o)
      (x := (w_393_0 ⟨o.val + 32, by omega⟩, w_393_1 ⟨o.val + 32, by omega⟩,
        w_393_2 ⟨o.val + 32, by omega⟩, w_393_3 ⟨o.val + 32, by omega⟩,
        w_393_4 ⟨o.val + 32, by omega⟩, w_393_5 ⟨o.val + 32, by omega⟩, ()))
    have b := prePlace287 (h := ph287_3 o)
      (x := (pw_219_0 ⟨o.val + 32, by omega⟩, pw_219_1 ⟨o.val + 32, by omega⟩,
        pw_219_2 ⟨o.val + 32, by omega⟩, pw_219_3 ⟨o.val + 32, by omega⟩, ()))
    simp only [Prod.mk.injEq] at a b
    obtain ⟨s0, s1, s2, s3⟩ := hs ⟨o.val + 32, by omega⟩
    exact ⟨by rw [a.1, b.1, s0], by rw [a.2.1, b.2.1, s1], by rw [a.2.2.1, b.2.2.1, s2],
      by rw [a.2.2.2.1, b.2.2.2.1, s3]⟩
  subst e_488_0 e_491_0 e_494_0 e_497_0 e_500_0 e_503_0 pe_290_0 pe_293_0 pe_296_0 pe_299_0
  have a := rootSum504 h486_0 h487_0 h489_0 h490_0 h492_0 h493_0 h495_0 h496_0 h498_0 h499_0 h501_0 h502_0 h504_5
  have b := prerootSum300 ph288_0 ph289_0 ph291_0 ph292_0 ph294_0 ph295_0 ph297_0 ph298_0 ph300_3
  simp only [Prod.mk.injEq] at a b
  rw [a.2.1, a.2.2.2.1, b.2.1, b.2.2.2.1]
  exact ⟨Finset.sum_congr rfl fun o _ ↦ (hp o).2.1, Finset.sum_congr rfl fun o _ ↦ (hp o).2.2.2⟩

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Fresh column `5`: the online root's member sums of keys are the preprocessing's. -/
theorem freshKeyCol5 {model : HashModel} {params : Stage_online_eval_0_0.Params} {params' : Stage_preprocess_0_0.Params}
    {w_393_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_4 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_5 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_507_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_507_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_507_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_507_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_507_4 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_507_5 : Fin 8 → ExactMatrix Q0 8 1 6}
    {w_508_0 : ExactMatrix Q0 8 1 6}
    {w_509_0 : ExactMatrix Q0 8 1 6}
    {w_510_0 : ExactMatrix Q0 8 1 6}
    {w_511_0 : ExactMatrix Q0 8 1 6}
    {w_512_0 : ExactMatrix Q0 8 1 6}
    {w_513_0 : ExactMatrix Q0 8 1 6}
    {w_514_0 : ExactMatrix Q0 8 1 6}
    {w_515_0 : ExactMatrix Q0 8 1 6}
    {w_516_0 : ExactMatrix Q0 8 1 6}
    {w_517_0 : ExactMatrix Q0 8 1 6}
    {w_518_0 : ExactMatrix Q0 8 1 6}
    {w_519_0 : ExactMatrix Q0 8 1 6}
    {w_520_0 : ExactMatrix Q0 8 1 6}
    {w_521_0 : ExactMatrix Q0 8 1 6}
    {w_522_0 : ExactMatrix Q0 8 1 6}
    {w_523_0 : ExactMatrix Q0 8 1 6}
    {w_524_0 : ExactMatrix Q0 8 1 6}
    {w_525_0 : ExactMatrix Q0 8 1 6}
    {w_526_0 : ExactMatrix Q0 8 1 6}
    {w_526_1 : ExactMatrix Q0 8 1 6}
    {w_526_2 : ExactMatrix Q0 8 1 6}
    {w_526_3 : ExactMatrix Q0 8 1 6}
    {w_526_4 : ExactMatrix Q0 8 1 6}
    {w_526_5 : ExactMatrix Q0 8 1 6}
    (e_510_0 : w_510_0 = matrixAdd w_508_0 w_509_0)
    (e_513_0 : w_513_0 = matrixAdd w_511_0 w_512_0)
    (e_516_0 : w_516_0 = matrixAdd w_514_0 w_515_0)
    (e_519_0 : w_519_0 = matrixAdd w_517_0 w_518_0)
    (e_522_0 : w_522_0 = matrixAdd w_520_0 w_521_0)
    (e_525_0 : w_525_0 = matrixAdd w_523_0 w_524_0)
    (h507_5 : (∀ i : Fin 8, Stage_online_eval_0_0.parallel_generatedRoot_507 Backend.backend model params i ((w_393_0 ⟨i.val + 40, by omega⟩), ((w_393_1 ⟨i.val + 40, by omega⟩), ((w_393_2 ⟨i.val + 40, by omega⟩), ((w_393_3 ⟨i.val + 40, by omega⟩), ((w_393_4 ⟨i.val + 40, by omega⟩), ((w_393_5 ⟨i.val + 40, by omega⟩), ())))))) ((w_507_0 i), ((w_507_1 i), ((w_507_2 i), ((w_507_3 i), ((w_507_4 i), ((w_507_5 i), ()))))))))
    (h508_0 : familyGetStatic w_507_0 (0) w_508_0)
    (h509_0 : familyGetStatic w_507_0 (1) w_509_0)
    (h511_0 : familyGetStatic w_507_1 (0) w_511_0)
    (h512_0 : familyGetStatic w_507_1 (1) w_512_0)
    (h514_0 : familyGetStatic w_507_2 (0) w_514_0)
    (h515_0 : familyGetStatic w_507_2 (1) w_515_0)
    (h517_0 : familyGetStatic w_507_3 (0) w_517_0)
    (h518_0 : familyGetStatic w_507_3 (1) w_518_0)
    (h520_0 : familyGetStatic w_507_4 (0) w_520_0)
    (h521_0 : familyGetStatic w_507_4 (1) w_521_0)
    (h523_0 : familyGetStatic w_507_5 (0) w_523_0)
    (h524_0 : familyGetStatic w_507_5 (1) w_524_0)
    (h526_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_online_eval_0_0.sequential_generatedRoot_526 Backend.backend model params i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_507_0, (w_507_1, (w_507_2, (w_507_3, (w_507_4, (w_507_5, ())))))))))))) next) (Int.toNat (6)) (w_510_0, (w_513_0, (w_516_0, (w_519_0, (w_522_0, (w_525_0, ())))))) (w_526_0, (w_526_1, (w_526_2, (w_526_3, (w_526_4, (w_526_5, ())))))))
    {pw_219_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_219_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {pw_303_0 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_303_1 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_303_2 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_303_3 : Fin 8 → ExactMatrix Q0 8 1 6}
    {pw_304_0 : ExactMatrix Q0 8 1 6}
    {pw_305_0 : ExactMatrix Q0 8 1 6}
    {pw_306_0 : ExactMatrix Q0 8 1 6}
    {pw_307_0 : ExactMatrix Q0 8 1 6}
    {pw_308_0 : ExactMatrix Q0 8 1 6}
    {pw_309_0 : ExactMatrix Q0 8 1 6}
    {pw_310_0 : ExactMatrix Q0 8 1 6}
    {pw_311_0 : ExactMatrix Q0 8 1 6}
    {pw_312_0 : ExactMatrix Q0 8 1 6}
    {pw_313_0 : ExactMatrix Q0 8 1 6}
    {pw_314_0 : ExactMatrix Q0 8 1 6}
    {pw_315_0 : ExactMatrix Q0 8 1 6}
    {pw_316_0 : ExactMatrix Q0 8 1 6}
    {pw_316_1 : ExactMatrix Q0 8 1 6}
    {pw_316_2 : ExactMatrix Q0 8 1 6}
    {pw_316_3 : ExactMatrix Q0 8 1 6}
    (pe_306_0 : pw_306_0 = matrixAdd pw_304_0 pw_305_0)
    (pe_309_0 : pw_309_0 = matrixAdd pw_307_0 pw_308_0)
    (pe_312_0 : pw_312_0 = matrixAdd pw_310_0 pw_311_0)
    (pe_315_0 : pw_315_0 = matrixAdd pw_313_0 pw_314_0)
    (ph303_3 : (∀ i : Fin 8, Stage_preprocess_0_0.parallel_generatedRoot_303 Backend.backend model params' i ((pw_219_0 ⟨i.val + 40, by omega⟩), ((pw_219_1 ⟨i.val + 40, by omega⟩), ((pw_219_2 ⟨i.val + 40, by omega⟩), ((pw_219_3 ⟨i.val + 40, by omega⟩), ())))) ((pw_303_0 i), ((pw_303_1 i), ((pw_303_2 i), ((pw_303_3 i), ()))))))
    (ph304_0 : familyGetStatic pw_303_0 (0) pw_304_0)
    (ph305_0 : familyGetStatic pw_303_0 (1) pw_305_0)
    (ph307_0 : familyGetStatic pw_303_1 (0) pw_307_0)
    (ph308_0 : familyGetStatic pw_303_1 (1) pw_308_0)
    (ph310_0 : familyGetStatic pw_303_2 (0) pw_310_0)
    (ph311_0 : familyGetStatic pw_303_2 (1) pw_311_0)
    (ph313_0 : familyGetStatic pw_303_3 (0) pw_313_0)
    (ph314_0 : familyGetStatic pw_303_3 (1) pw_314_0)
    (ph316_3 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => Stage_preprocess_0_0.sequential_generatedRoot_316 Backend.backend model params' i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (pw_303_0, (pw_303_1, (pw_303_2, (pw_303_3, ())))))))) next) (Int.toNat (6)) (pw_306_0, (pw_309_0, (pw_312_0, (pw_315_0, ())))) (pw_316_0, (pw_316_1, (pw_316_2, (pw_316_3, ())))))
    (hs : ∀ o : Fin 48, w_393_0 o = pw_219_0 o ∧ w_393_1 o = pw_219_1 o ∧ w_393_2 o = pw_219_2 o ∧
      w_393_3 o = pw_219_3 o) :
    w_526_1 = pw_316_1 ∧ w_526_3 = pw_316_3 := by
  have hp : ∀ o : Fin 8, w_507_0 o = pw_303_0 o ∧ w_507_1 o = pw_303_1 o ∧ w_507_2 o = pw_303_2 o ∧
      w_507_3 o = pw_303_3 o := by
    intro o
    have a := place507 (h := h507_5 o)
      (x := (w_393_0 ⟨o.val + 40, by omega⟩, w_393_1 ⟨o.val + 40, by omega⟩,
        w_393_2 ⟨o.val + 40, by omega⟩, w_393_3 ⟨o.val + 40, by omega⟩,
        w_393_4 ⟨o.val + 40, by omega⟩, w_393_5 ⟨o.val + 40, by omega⟩, ()))
    have b := prePlace303 (h := ph303_3 o)
      (x := (pw_219_0 ⟨o.val + 40, by omega⟩, pw_219_1 ⟨o.val + 40, by omega⟩,
        pw_219_2 ⟨o.val + 40, by omega⟩, pw_219_3 ⟨o.val + 40, by omega⟩, ()))
    simp only [Prod.mk.injEq] at a b
    obtain ⟨s0, s1, s2, s3⟩ := hs ⟨o.val + 40, by omega⟩
    exact ⟨by rw [a.1, b.1, s0], by rw [a.2.1, b.2.1, s1], by rw [a.2.2.1, b.2.2.1, s2],
      by rw [a.2.2.2.1, b.2.2.2.1, s3]⟩
  subst e_510_0 e_513_0 e_516_0 e_519_0 e_522_0 e_525_0 pe_306_0 pe_309_0 pe_312_0 pe_315_0
  have a := rootSum526 h508_0 h509_0 h511_0 h512_0 h514_0 h515_0 h517_0 h518_0 h520_0 h521_0 h523_0 h524_0 h526_5
  have b := prerootSum316 ph304_0 ph305_0 ph307_0 ph308_0 ph310_0 ph311_0 ph313_0 ph314_0 ph316_3
  simp only [Prod.mk.injEq] at a b
  rw [a.2.1, a.2.2.2.1, b.2.1, b.2.2.2.1]
  exact ⟨Finset.sum_congr rfl fun o _ ↦ (hp o).2.1, Finset.sum_congr rfl fun o _ ↦ (hp o).2.2.2⟩

end RefreshTopProof.Part_0_0
