import Part_I_Local
import TopInst
import Part_I_Fresh
import Part_I_FreshCarrier
import Part_I_RootSums
import TopAction
import TopChain

/-! Chain theorems of the fresh noise of part `I` (written by `generator/gen_fresh_chain.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof Mxx.ExponentLut.Refresh

/-- The bound of a sampled member of the fresh PRF. -/
abbrev freshSampleBound (i0 : Nat) (Ba Bs BB BB' Bh5 BS Bh0 : Nat) : Nat :=
  ((3 * lutBranchBound393 i0 (freshReadoutBound Ba Bs BB BB' Bh5 BS) Bh0 + ((6 + 6) * N * Bh0 * 16384 + (6 + 6) * N * (freshReadoutBound Ba Bs BB BB' Bh5 BS) * 16384 + (N * 1 * ((qd i0) / 2) + (qd i0) / 2))))

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Every member of the fresh PRF, read out and sampled into its LUT branches. -/
theorem freshSamples {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params}
    {w_358_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_359_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_360_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_361_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
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
    (h386_5 : (∀ i : Fin 240, W_Stage_online_eval.parallel_parallel_generatedRoot_290_388 Backend.backend model params i0 i (w_371_0, (w_372_0, (w_373_0, (w_374_0, (w_375_0, (w_376_0, (w_377_0, (w_378_0, (w_379_0, (w_380_0, (w_381_0, (w_382_0, (w_383_0, (w_363_0, ((w_384_0 i), ((w_385_0 i), ())))))))))))))))) ((w_386_0 i), ((w_386_1 i), ((w_386_2 i), ((w_386_3 i), ((w_386_4 i), ((w_386_5 i), ()))))))))
    (h387_5 : (∀ i : Fin 48, W_Stage_online_eval.parallel_parallel_generatedRoot_290_389 Backend.backend model params i0 i (w_386_0, (w_386_1, (w_386_2, (w_386_3, (w_386_4, (w_386_5, ())))))) ((w_387_0 i), ((w_387_1 i), ((w_387_2 i), ((w_387_3 i), ((w_387_4 i), ((w_387_5 i), ()))))))))
    (h390_5 : (∀ i : Fin 240, W_Stage_online_eval.parallel_parallel_generatedRoot_290_392 Backend.backend model params i0 i (w_371_0, (w_372_0, (w_373_0, (w_374_0, (w_375_0, (w_376_0, (w_387_0, (w_387_2, (w_387_4, (w_380_0, (w_387_1, (w_387_3, (w_387_5, (w_363_0, ((w_388_0 i), ((w_389_0 i), ())))))))))))))))) ((w_390_0 i), ((w_390_1 i), ((w_390_2 i), ((w_390_3 i), ((w_390_4 i), ((w_390_5 i), ()))))))))
    (h391_5 : (∀ i : Fin 48, W_Stage_online_eval.parallel_parallel_generatedRoot_290_393 Backend.backend model params i0 i (w_390_0, (w_390_1, (w_390_2, (w_390_3, (w_390_4, (w_390_5, ())))))) ((w_391_0 i), ((w_391_1 i), ((w_391_2 i), ((w_391_3 i), ((w_391_4 i), ((w_391_5 i), ()))))))))
    (h392_5 : (∀ i : Fin 48, W_Stage_online_eval.parallel_parallel_generatedRoot_290_394 Backend.backend model params i0 i (w_365_0, (w_366_0, (w_367_0, (w_368_0, (w_369_0, (w_363_0, (w_370_0, ((w_391_4 i), ((w_391_0 i), ((w_391_2 i), ((w_391_5 i), ((w_391_1 i), ((w_391_3 i), ()))))))))))))) ((w_392_0 i), ((w_392_1 i), ((w_392_2 i), ((w_392_3 i), ((w_392_4 i), ((w_392_5 i), ()))))))))
    (h393_5 : (∀ i : Fin 48, W_Stage_online_eval.parallel_parallel_generatedRoot_290_395 Backend.backend model params i0 i (w_358_0, (w_359_0, (w_360_0, (w_361_0, (w_362_0, (w_363_0, (w_364_0, ((w_392_4 i), ((w_392_0 i), ((w_392_2 i), ((w_392_5 i), ((w_392_1 i), ((w_392_3 i), ()))))))))))))) ((w_393_0 i), ((w_393_1 i), ((w_393_2 i), ((w_393_3 i), ((w_393_4 i), ((w_393_5 i), ()))))))))
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    {g6' : ExactMatrix Q6 N 1 12} {g7 : ExactMatrix Q7 N 1 12}
    (hg6' : gadgetMatrixRuns Backend.backend 32768 12 g6') (hg7 : centeredRebaseRuns g6' g7)
    {Bs : Nat}
    (hone : Rel sm ![![w_377_0, w_381_0], ![w_378_0, w_382_0]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * 1 * unitPair sp j) • g7) ![w_379_0, w_383_0] Bs)
    {g5' : ExactMatrix Q5 N 1 10} {h6 : ExactMatrix Q6 N 1 10}
    (hg5' : gadgetMatrixRuns Backend.backend 32768 10 g5') (hh6 : centeredRebaseRuns g5' h6)
    {g0' : ExactMatrix Q0 N 1 8} {h5 : ExactMatrix Q5 N 1 8}
    (hg0' : gadgetMatrixRuns Backend.backend 32768 8 g0') (hh5 : centeredRebaseRuns g0' h5)
    {a : Fin 5 → Nat} {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 32768 14 G)
    {Ba : Nat}
    (ha : ∀ k : Fin 5, Rel sm ![![w_372_0 k, w_375_0 k], ![w_373_0 k, w_376_0 k]]
      (fun j ↦ reducePoly Q7 N (Xe (n := N) (2 * a k) * unitPair sm j) • G) ![w_371_0 k, w_374_0 k] Ba)
    (hpay : ∀ k : Fin 5, w_380_0 k = fun _ _ ↦ reducePoly Q7 N (Xe (n := N) (2 * a k)))
    {s : Fin 240 → Int} {eB : Fin 240 → Fin 2 → ErrorMatrix N 1 10} {BB : Nat}
    (hrhs : ∀ (i : Fin 240) j, reducePoly Q6 N sp • rhsTop model w_363_0 Q6 10 (freshRhs0 i0) i j +
        ![w_384_0 i, w_385_0 i] j =
      reducePoly Q6 N (((p5 : Nat) : Int) * ((s i : Int) : ErrorPoly N) * unitPair sp j) • h6 +
        reduceMatrix Q6 N 1 10 (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    {s' : Fin 240 → Int} {eB' : Fin 240 → Fin 2 → ErrorMatrix N 1 8} {BB' : Nat}
    (hrhs' : ∀ (i : Fin 240) j, reducePoly Q5 N sp • rhsTop model w_363_0 Q5 8 (freshRhs1 i0) i j +
        ![w_388_0 i, w_389_0 i] j =
      reducePoly Q5 N (((p4 : Nat) : Int) * ((s' i : Int) : ErrorPoly N) * unitPair sp j) • h5 +
        reduceMatrix Q5 N 1 8 (eB' i j))
    (heB' : ∀ i j, CoeffBound (eB' i j) BB')
    {sel sel' : Fin 48 → Fin 5}
    (hsel : ∀ o (k : Fin 5), s (cellIndex o k) = if k = sel o then 1 else 0)
    (hsel' : ∀ o (k : Fin 5), s' (cellIndex o k) = if k = sel' o then 1 else 0)
    {Bh5 : Nat} (hhelp5 : ∀ i : Fin 4, Rel sm (lutA model w_363_0 Q5 8 2 (sigma i))
      (fun j ↦ reducePoly Q5 N (((p4 : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • h5) ![w_369_0 i, w_370_0 i] Bh5)
    {g1' : ExactMatrix (R i0) N 1 6} {g0 : ExactMatrix Q0 N 1 6}
    (hg1' : gadgetMatrixRuns Backend.backend 32768 6 g1') (hg0 : centeredRebaseRuns g1' g0)
    {inverse : Nat} (hinverse : (inverse : ExactPoly Q0 N) * ((4 : Nat) : ExactPoly Q0 N) = 1)
    {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly Q0 N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_365_0, w_366_0], ![w_367_0, w_368_0]] 0 j i + ![![w_365_0, w_366_0], ![w_367_0, w_368_0]] 1 j i =
        (lutCoefficient Q0 inverse (fun k ↦ reducePoly Q0 N (readoutTable k)) (sigma i).toNat *
          reducePoly Q0 N (((((qd i0) : Nat) : Int) : ErrorPoly N) * unitPair sp j)) • g0 +
            reduceMatrix Q0 N 1 6 (e j))
    {Bh0 : Nat} (hhelp0 : ∀ i : Fin 4, Rel sm (lutA model w_363_0 Q0 6 3 (sigma i))
      (fun j ↦ reducePoly Q0 N ((((qd i0) : Nat) : Int) *
        unitPair (substitution (sigma i).toNat (sigma_odd i) sm) j) • g0) ![w_362_0 i, w_364_0 i] Bh0)
    (o : Fin 48) :
    Rel sm ![![w_393_0 o, w_393_1 o], ![w_393_2 o, w_393_3 o]]
      (fun j ↦ (∑ i : Fin 4, scaleUpMatrix Q0 (qd i0)
        (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i)
          (Xe (2 * readoutExp (a (sel' o) + a (sel o))))) •
          (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
              ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i))))
      ![w_393_4 o, w_393_5 o] ((freshSampleBound i0) Ba Bs BB BB' Bh5 BS Bh0) := by
  have hread := freshReadout (model := model) (h386 := h386_5) (h387 := h387_5) (h390 := h390_5)
    (h391 := h391_5) (h392 := h392_5) (hsm := hsm) (hG := hG) (hg6' := hg6') (hg7 := hg7)
    (hg5' := hg5') (hh6 := hh6) (ha := ha) (hpay := hpay) (hone := hone) (hrhs := hrhs) (heB := heB)
    (hrhs' := hrhs') (heB' := heB') (hsel := hsel) (hsel' := hsel') (hg0' := hg0') (hh5 := hh5)
    (hhelp := hhelp5) (hinverse := hinverse) (hswitch := hswitch) (o := o)
  exact lut393 (h := h393_5 o) hsm hg1' hg0 hread hhelp0


set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Column `0` of the fresh noise times the digits of `Q_q`. -/
theorem freshCol0 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params}
    {w_358_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_359_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_360_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_361_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
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
    {w_414_0 : ExactMatrix (R i0) 8 1 6}
    {w_415_0 : ExactMatrix (R i0) 8 1 1}
    {w_416_0 : ExactMatrix (R i0) 8 6 1}
    {w_417_0 : ExactMatrix (R i0) 8 6 1}
    {w_418_0 : ExactMatrix (R i0) 8 1 1}
    (e_397_0 : w_397_0 = matrixAdd w_395_0 w_396_0)
    (e_400_0 : w_400_0 = matrixAdd w_398_0 w_399_0)
    (e_403_0 : w_403_0 = matrixAdd w_401_0 w_402_0)
    (e_406_0 : w_406_0 = matrixAdd w_404_0 w_405_0)
    (e_409_0 : w_409_0 = matrixAdd w_407_0 w_408_0)
    (e_412_0 : w_412_0 = matrixAdd w_410_0 w_411_0)
    (e_415_0 : w_415_0 = (matrixPolynomial [((QqZ (i0 : Int)))] : ExactMatrix (R i0) 8 1 1))
    (e_418_0 : w_418_0 = matrixMul w_414_0 w_417_0)
    (h394_5 : (∀ i : Fin 8, W_Stage_online_eval.parallel_parallel_generatedRoot_290_396 Backend.backend model params i0 i ((w_393_0 ⟨i.val, by omega⟩), ((w_393_1 ⟨i.val, by omega⟩), ((w_393_2 ⟨i.val, by omega⟩), ((w_393_3 ⟨i.val, by omega⟩), ((w_393_4 ⟨i.val, by omega⟩), ((w_393_5 ⟨i.val, by omega⟩), ())))))) ((w_394_0 i), ((w_394_1 i), ((w_394_2 i), ((w_394_3 i), ((w_394_4 i), ((w_394_5 i), ()))))))))
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
    (h413_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => W_Stage_online_eval.sequential_parallel_generatedRoot_290_415 Backend.backend model params i0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_394_0, (w_394_1, (w_394_2, (w_394_3, (w_394_4, (w_394_5, ())))))))))))) next) (Int.toNat (6)) (w_397_0, (w_400_0, (w_403_0, (w_406_0, (w_409_0, (w_412_0, ())))))) (w_413_0, (w_413_1, (w_413_2, (w_413_3, (w_413_4, (w_413_5, ())))))))
    (h414_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_413_5 w_414_0)
    (h416_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_415_0 w_416_0)
    (h417_0 : compactCenteredRebaseRuns 16384 w_416_0 w_417_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {ζ : Fin 48 → Nat} {Bs : Nat}
    (hsamp : ∀ o : Fin 48, Rel sm ![![w_393_0 o, w_393_1 o], ![w_393_2 o, w_393_3 o]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 (qd i0)
        (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ o))) •
          (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
              ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i)))
      ![w_393_4 o, w_393_5 o] Bs)
    {T : Fin 4 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix (R i0) N 1 6}
    (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly (R i0) N) * ((4 : Nat) : ExactPoly (R i0) N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i =
        (lutCoefficient (R i0) inverse (fun k ↦ reducePoly (R i0) N (T k)) (sigma i).toNat *
          reducePoly (R i0) N (Δ' * unitPair sp j)) • gq + reduceMatrix (R i0) N 1 6 (e j)) :
    ∃ e : ErrorMatrix N 1 1,
      CoeffBound e (6 * N * ((8 * Bs + (N + 2) * ((qd i0) / 2)) / (qd i0) + 32 * (N * 1 * BS)) * 16384) ∧
      w_418_0 = maskPart (reducePoly (R i0) N sm) (fun k ↦
          blockModSwitch [q0, q1, q2, q3] 1 (![![w_413_0, w_413_1], ![w_413_2, w_413_3]] k 1) * w_417_0) -
        reducePoly (R i0) N (Δ' * (∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val, by omega⟩)) *
          unitPair sp 1) • (matrixPolynomial [(QqZ (i0 : Int))] : ExactMatrix (R i0) N 1 1) +
        reduceMatrix (R i0) N 1 1 e := by
  subst e_397_0 e_400_0 e_403_0 e_406_0 e_409_0 e_412_0
  have hs := rootSum413 h395_0 h396_0 h398_0 h399_0 h401_0 h402_0 h404_0 h405_0 h407_0 h408_0 h410_0 h411_0 h413_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  have hc := freshColumn0 (model := model) (hplace := h394_5) hsm (ζ := ζ) hsamp hinverse hswitch
    (by rw [← s5]; exact h414_0)
  rw [← s0, ← s1, ← s2, ← s3] at hc
  obtain ⟨e, he, hrow⟩ := Rel.row hc 1
  simp only [Matrix.cons_val_one, Matrix.head_cons] at hrow
  rw [e_418_0]
  exact row_mul_qqQ hgq he hrow (by rw [← e_415_0]; exact h416_0) h417_0


set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Column `1` of the fresh noise times the digits of `Q_q`. -/
theorem freshCol1 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params}
    {w_358_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_359_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_360_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_361_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_393_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_4 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_5 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_415_0 : ExactMatrix (R i0) 8 1 1}
    {w_416_0 : ExactMatrix (R i0) 8 6 1}
    {w_417_0 : ExactMatrix (R i0) 8 6 1}
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
    {w_439_0 : ExactMatrix (R i0) 8 1 6}
    {w_440_0 : ExactMatrix (R i0) 8 1 1}
    (e_415_0 : w_415_0 = (matrixPolynomial [((QqZ (i0 : Int)))] : ExactMatrix (R i0) 8 1 1))
    (e_422_0 : w_422_0 = matrixAdd w_420_0 w_421_0)
    (e_425_0 : w_425_0 = matrixAdd w_423_0 w_424_0)
    (e_428_0 : w_428_0 = matrixAdd w_426_0 w_427_0)
    (e_431_0 : w_431_0 = matrixAdd w_429_0 w_430_0)
    (e_434_0 : w_434_0 = matrixAdd w_432_0 w_433_0)
    (e_437_0 : w_437_0 = matrixAdd w_435_0 w_436_0)
    (e_440_0 : w_440_0 = matrixMul w_439_0 w_417_0)
    (h419_5 : (∀ i : Fin 8, W_Stage_online_eval.parallel_parallel_generatedRoot_290_421 Backend.backend model params i0 i ((w_393_0 ⟨i.val + 8, by omega⟩), ((w_393_1 ⟨i.val + 8, by omega⟩), ((w_393_2 ⟨i.val + 8, by omega⟩), ((w_393_3 ⟨i.val + 8, by omega⟩), ((w_393_4 ⟨i.val + 8, by omega⟩), ((w_393_5 ⟨i.val + 8, by omega⟩), ())))))) ((w_419_0 i), ((w_419_1 i), ((w_419_2 i), ((w_419_3 i), ((w_419_4 i), ((w_419_5 i), ()))))))))
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
    (h438_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => W_Stage_online_eval.sequential_parallel_generatedRoot_290_440 Backend.backend model params i0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_419_0, (w_419_1, (w_419_2, (w_419_3, (w_419_4, (w_419_5, ())))))))))))) next) (Int.toNat (6)) (w_422_0, (w_425_0, (w_428_0, (w_431_0, (w_434_0, (w_437_0, ())))))) (w_438_0, (w_438_1, (w_438_2, (w_438_3, (w_438_4, (w_438_5, ())))))))
    (h439_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_438_5 w_439_0)
    (h416_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_415_0 w_416_0)
    (h417_0 : compactCenteredRebaseRuns 16384 w_416_0 w_417_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {ζ : Fin 48 → Nat} {Bs : Nat}
    (hsamp : ∀ o : Fin 48, Rel sm ![![w_393_0 o, w_393_1 o], ![w_393_2 o, w_393_3 o]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 (qd i0)
        (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ o))) •
          (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
              ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i)))
      ![w_393_4 o, w_393_5 o] Bs)
    {T : Fin 4 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix (R i0) N 1 6}
    (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly (R i0) N) * ((4 : Nat) : ExactPoly (R i0) N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i =
        (lutCoefficient (R i0) inverse (fun k ↦ reducePoly (R i0) N (T k)) (sigma i).toNat *
          reducePoly (R i0) N (Δ' * unitPair sp j)) • gq + reduceMatrix (R i0) N 1 6 (e j)) :
    ∃ e : ErrorMatrix N 1 1,
      CoeffBound e (6 * N * ((8 * Bs + (N + 2) * ((qd i0) / 2)) / (qd i0) + 32 * (N * 1 * BS)) * 16384) ∧
      w_440_0 = maskPart (reducePoly (R i0) N sm) (fun k ↦
          blockModSwitch [q0, q1, q2, q3] 1 (![![w_438_0, w_438_1], ![w_438_2, w_438_3]] k 1) * w_417_0) -
        reducePoly (R i0) N (Δ' * (∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val + 8, by omega⟩)) *
          unitPair sp 1) • (matrixPolynomial [(QqZ (i0 : Int))] : ExactMatrix (R i0) N 1 1) +
        reduceMatrix (R i0) N 1 1 e := by
  subst e_422_0 e_425_0 e_428_0 e_431_0 e_434_0 e_437_0
  have hs := rootSum438 h420_0 h421_0 h423_0 h424_0 h426_0 h427_0 h429_0 h430_0 h432_0 h433_0 h435_0 h436_0 h438_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  have hc := freshColumn1 (model := model) (hplace := h419_5) hsm (ζ := ζ) hsamp hinverse hswitch
    (by rw [← s5]; exact h439_0)
  rw [← s0, ← s1, ← s2, ← s3] at hc
  obtain ⟨e, he, hrow⟩ := Rel.row hc 1
  simp only [Matrix.cons_val_one, Matrix.head_cons] at hrow
  rw [e_440_0]
  exact row_mul_qqQ hgq he hrow (by rw [← e_415_0]; exact h416_0) h417_0


set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Column `2` of the fresh noise times the digits of `Q_q`. -/
theorem freshCol2 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params}
    {w_358_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_359_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_360_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_361_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_393_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_4 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_5 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_415_0 : ExactMatrix (R i0) 8 1 1}
    {w_416_0 : ExactMatrix (R i0) 8 6 1}
    {w_417_0 : ExactMatrix (R i0) 8 6 1}
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
    {w_461_0 : ExactMatrix (R i0) 8 1 6}
    {w_462_0 : ExactMatrix (R i0) 8 1 1}
    (e_415_0 : w_415_0 = (matrixPolynomial [((QqZ (i0 : Int)))] : ExactMatrix (R i0) 8 1 1))
    (e_444_0 : w_444_0 = matrixAdd w_442_0 w_443_0)
    (e_447_0 : w_447_0 = matrixAdd w_445_0 w_446_0)
    (e_450_0 : w_450_0 = matrixAdd w_448_0 w_449_0)
    (e_453_0 : w_453_0 = matrixAdd w_451_0 w_452_0)
    (e_456_0 : w_456_0 = matrixAdd w_454_0 w_455_0)
    (e_459_0 : w_459_0 = matrixAdd w_457_0 w_458_0)
    (e_462_0 : w_462_0 = matrixMul w_461_0 w_417_0)
    (h441_5 : (∀ i : Fin 8, W_Stage_online_eval.parallel_parallel_generatedRoot_290_443 Backend.backend model params i0 i ((w_393_0 ⟨i.val + 16, by omega⟩), ((w_393_1 ⟨i.val + 16, by omega⟩), ((w_393_2 ⟨i.val + 16, by omega⟩), ((w_393_3 ⟨i.val + 16, by omega⟩), ((w_393_4 ⟨i.val + 16, by omega⟩), ((w_393_5 ⟨i.val + 16, by omega⟩), ())))))) ((w_441_0 i), ((w_441_1 i), ((w_441_2 i), ((w_441_3 i), ((w_441_4 i), ((w_441_5 i), ()))))))))
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
    (h460_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => W_Stage_online_eval.sequential_parallel_generatedRoot_290_462 Backend.backend model params i0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_441_0, (w_441_1, (w_441_2, (w_441_3, (w_441_4, (w_441_5, ())))))))))))) next) (Int.toNat (6)) (w_444_0, (w_447_0, (w_450_0, (w_453_0, (w_456_0, (w_459_0, ())))))) (w_460_0, (w_460_1, (w_460_2, (w_460_3, (w_460_4, (w_460_5, ())))))))
    (h461_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_460_5 w_461_0)
    (h416_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_415_0 w_416_0)
    (h417_0 : compactCenteredRebaseRuns 16384 w_416_0 w_417_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {ζ : Fin 48 → Nat} {Bs : Nat}
    (hsamp : ∀ o : Fin 48, Rel sm ![![w_393_0 o, w_393_1 o], ![w_393_2 o, w_393_3 o]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 (qd i0)
        (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ o))) •
          (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
              ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i)))
      ![w_393_4 o, w_393_5 o] Bs)
    {T : Fin 4 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix (R i0) N 1 6}
    (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly (R i0) N) * ((4 : Nat) : ExactPoly (R i0) N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i =
        (lutCoefficient (R i0) inverse (fun k ↦ reducePoly (R i0) N (T k)) (sigma i).toNat *
          reducePoly (R i0) N (Δ' * unitPair sp j)) • gq + reduceMatrix (R i0) N 1 6 (e j)) :
    ∃ e : ErrorMatrix N 1 1,
      CoeffBound e (6 * N * ((8 * Bs + (N + 2) * ((qd i0) / 2)) / (qd i0) + 32 * (N * 1 * BS)) * 16384) ∧
      w_462_0 = maskPart (reducePoly (R i0) N sm) (fun k ↦
          blockModSwitch [q0, q1, q2, q3] 1 (![![w_460_0, w_460_1], ![w_460_2, w_460_3]] k 1) * w_417_0) -
        reducePoly (R i0) N (Δ' * (∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val + 16, by omega⟩)) *
          unitPair sp 1) • (matrixPolynomial [(QqZ (i0 : Int))] : ExactMatrix (R i0) N 1 1) +
        reduceMatrix (R i0) N 1 1 e := by
  subst e_444_0 e_447_0 e_450_0 e_453_0 e_456_0 e_459_0
  have hs := rootSum460 h442_0 h443_0 h445_0 h446_0 h448_0 h449_0 h451_0 h452_0 h454_0 h455_0 h457_0 h458_0 h460_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  have hc := freshColumn2 (model := model) (hplace := h441_5) hsm (ζ := ζ) hsamp hinverse hswitch
    (by rw [← s5]; exact h461_0)
  rw [← s0, ← s1, ← s2, ← s3] at hc
  obtain ⟨e, he, hrow⟩ := Rel.row hc 1
  simp only [Matrix.cons_val_one, Matrix.head_cons] at hrow
  rw [e_462_0]
  exact row_mul_qqQ hgq he hrow (by rw [← e_415_0]; exact h416_0) h417_0


set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Column `3` of the fresh noise times the digits of `Q_q`. -/
theorem freshCol3 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params}
    {w_358_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_359_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_360_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_361_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_393_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_4 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_5 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_415_0 : ExactMatrix (R i0) 8 1 1}
    {w_416_0 : ExactMatrix (R i0) 8 6 1}
    {w_417_0 : ExactMatrix (R i0) 8 6 1}
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
    {w_483_0 : ExactMatrix (R i0) 8 1 6}
    {w_484_0 : ExactMatrix (R i0) 8 1 1}
    (e_415_0 : w_415_0 = (matrixPolynomial [((QqZ (i0 : Int)))] : ExactMatrix (R i0) 8 1 1))
    (e_466_0 : w_466_0 = matrixAdd w_464_0 w_465_0)
    (e_469_0 : w_469_0 = matrixAdd w_467_0 w_468_0)
    (e_472_0 : w_472_0 = matrixAdd w_470_0 w_471_0)
    (e_475_0 : w_475_0 = matrixAdd w_473_0 w_474_0)
    (e_478_0 : w_478_0 = matrixAdd w_476_0 w_477_0)
    (e_481_0 : w_481_0 = matrixAdd w_479_0 w_480_0)
    (e_484_0 : w_484_0 = matrixMul w_483_0 w_417_0)
    (h463_5 : (∀ i : Fin 8, W_Stage_online_eval.parallel_parallel_generatedRoot_290_465 Backend.backend model params i0 i ((w_393_0 ⟨i.val + 24, by omega⟩), ((w_393_1 ⟨i.val + 24, by omega⟩), ((w_393_2 ⟨i.val + 24, by omega⟩), ((w_393_3 ⟨i.val + 24, by omega⟩), ((w_393_4 ⟨i.val + 24, by omega⟩), ((w_393_5 ⟨i.val + 24, by omega⟩), ())))))) ((w_463_0 i), ((w_463_1 i), ((w_463_2 i), ((w_463_3 i), ((w_463_4 i), ((w_463_5 i), ()))))))))
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
    (h482_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => W_Stage_online_eval.sequential_parallel_generatedRoot_290_484 Backend.backend model params i0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_463_0, (w_463_1, (w_463_2, (w_463_3, (w_463_4, (w_463_5, ())))))))))))) next) (Int.toNat (6)) (w_466_0, (w_469_0, (w_472_0, (w_475_0, (w_478_0, (w_481_0, ())))))) (w_482_0, (w_482_1, (w_482_2, (w_482_3, (w_482_4, (w_482_5, ())))))))
    (h483_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_482_5 w_483_0)
    (h416_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_415_0 w_416_0)
    (h417_0 : compactCenteredRebaseRuns 16384 w_416_0 w_417_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {ζ : Fin 48 → Nat} {Bs : Nat}
    (hsamp : ∀ o : Fin 48, Rel sm ![![w_393_0 o, w_393_1 o], ![w_393_2 o, w_393_3 o]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 (qd i0)
        (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ o))) •
          (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
              ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i)))
      ![w_393_4 o, w_393_5 o] Bs)
    {T : Fin 4 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix (R i0) N 1 6}
    (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly (R i0) N) * ((4 : Nat) : ExactPoly (R i0) N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i =
        (lutCoefficient (R i0) inverse (fun k ↦ reducePoly (R i0) N (T k)) (sigma i).toNat *
          reducePoly (R i0) N (Δ' * unitPair sp j)) • gq + reduceMatrix (R i0) N 1 6 (e j)) :
    ∃ e : ErrorMatrix N 1 1,
      CoeffBound e (6 * N * ((8 * Bs + (N + 2) * ((qd i0) / 2)) / (qd i0) + 32 * (N * 1 * BS)) * 16384) ∧
      w_484_0 = maskPart (reducePoly (R i0) N sm) (fun k ↦
          blockModSwitch [q0, q1, q2, q3] 1 (![![w_482_0, w_482_1], ![w_482_2, w_482_3]] k 1) * w_417_0) -
        reducePoly (R i0) N (Δ' * (∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val + 24, by omega⟩)) *
          unitPair sp 1) • (matrixPolynomial [(QqZ (i0 : Int))] : ExactMatrix (R i0) N 1 1) +
        reduceMatrix (R i0) N 1 1 e := by
  subst e_466_0 e_469_0 e_472_0 e_475_0 e_478_0 e_481_0
  have hs := rootSum482 h464_0 h465_0 h467_0 h468_0 h470_0 h471_0 h473_0 h474_0 h476_0 h477_0 h479_0 h480_0 h482_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  have hc := freshColumn3 (model := model) (hplace := h463_5) hsm (ζ := ζ) hsamp hinverse hswitch
    (by rw [← s5]; exact h483_0)
  rw [← s0, ← s1, ← s2, ← s3] at hc
  obtain ⟨e, he, hrow⟩ := Rel.row hc 1
  simp only [Matrix.cons_val_one, Matrix.head_cons] at hrow
  rw [e_484_0]
  exact row_mul_qqQ hgq he hrow (by rw [← e_415_0]; exact h416_0) h417_0


set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Column `4` of the fresh noise times the digits of `Q_q`. -/
theorem freshCol4 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params}
    {w_358_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_359_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_360_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_361_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_393_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_4 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_5 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_415_0 : ExactMatrix (R i0) 8 1 1}
    {w_416_0 : ExactMatrix (R i0) 8 6 1}
    {w_417_0 : ExactMatrix (R i0) 8 6 1}
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
    {w_505_0 : ExactMatrix (R i0) 8 1 6}
    {w_506_0 : ExactMatrix (R i0) 8 1 1}
    (e_415_0 : w_415_0 = (matrixPolynomial [((QqZ (i0 : Int)))] : ExactMatrix (R i0) 8 1 1))
    (e_488_0 : w_488_0 = matrixAdd w_486_0 w_487_0)
    (e_491_0 : w_491_0 = matrixAdd w_489_0 w_490_0)
    (e_494_0 : w_494_0 = matrixAdd w_492_0 w_493_0)
    (e_497_0 : w_497_0 = matrixAdd w_495_0 w_496_0)
    (e_500_0 : w_500_0 = matrixAdd w_498_0 w_499_0)
    (e_503_0 : w_503_0 = matrixAdd w_501_0 w_502_0)
    (e_506_0 : w_506_0 = matrixMul w_505_0 w_417_0)
    (h485_5 : (∀ i : Fin 8, W_Stage_online_eval.parallel_parallel_generatedRoot_290_487 Backend.backend model params i0 i ((w_393_0 ⟨i.val + 32, by omega⟩), ((w_393_1 ⟨i.val + 32, by omega⟩), ((w_393_2 ⟨i.val + 32, by omega⟩), ((w_393_3 ⟨i.val + 32, by omega⟩), ((w_393_4 ⟨i.val + 32, by omega⟩), ((w_393_5 ⟨i.val + 32, by omega⟩), ())))))) ((w_485_0 i), ((w_485_1 i), ((w_485_2 i), ((w_485_3 i), ((w_485_4 i), ((w_485_5 i), ()))))))))
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
    (h504_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => W_Stage_online_eval.sequential_parallel_generatedRoot_290_506 Backend.backend model params i0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_485_0, (w_485_1, (w_485_2, (w_485_3, (w_485_4, (w_485_5, ())))))))))))) next) (Int.toNat (6)) (w_488_0, (w_491_0, (w_494_0, (w_497_0, (w_500_0, (w_503_0, ())))))) (w_504_0, (w_504_1, (w_504_2, (w_504_3, (w_504_4, (w_504_5, ())))))))
    (h505_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_504_5 w_505_0)
    (h416_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_415_0 w_416_0)
    (h417_0 : compactCenteredRebaseRuns 16384 w_416_0 w_417_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {ζ : Fin 48 → Nat} {Bs : Nat}
    (hsamp : ∀ o : Fin 48, Rel sm ![![w_393_0 o, w_393_1 o], ![w_393_2 o, w_393_3 o]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 (qd i0)
        (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ o))) •
          (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
              ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i)))
      ![w_393_4 o, w_393_5 o] Bs)
    {T : Fin 4 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix (R i0) N 1 6}
    (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly (R i0) N) * ((4 : Nat) : ExactPoly (R i0) N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i =
        (lutCoefficient (R i0) inverse (fun k ↦ reducePoly (R i0) N (T k)) (sigma i).toNat *
          reducePoly (R i0) N (Δ' * unitPair sp j)) • gq + reduceMatrix (R i0) N 1 6 (e j)) :
    ∃ e : ErrorMatrix N 1 1,
      CoeffBound e (6 * N * ((8 * Bs + (N + 2) * ((qd i0) / 2)) / (qd i0) + 32 * (N * 1 * BS)) * 16384) ∧
      w_506_0 = maskPart (reducePoly (R i0) N sm) (fun k ↦
          blockModSwitch [q0, q1, q2, q3] 1 (![![w_504_0, w_504_1], ![w_504_2, w_504_3]] k 1) * w_417_0) -
        reducePoly (R i0) N (Δ' * (∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val + 32, by omega⟩)) *
          unitPair sp 1) • (matrixPolynomial [(QqZ (i0 : Int))] : ExactMatrix (R i0) N 1 1) +
        reduceMatrix (R i0) N 1 1 e := by
  subst e_488_0 e_491_0 e_494_0 e_497_0 e_500_0 e_503_0
  have hs := rootSum504 h486_0 h487_0 h489_0 h490_0 h492_0 h493_0 h495_0 h496_0 h498_0 h499_0 h501_0 h502_0 h504_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  have hc := freshColumn4 (model := model) (hplace := h485_5) hsm (ζ := ζ) hsamp hinverse hswitch
    (by rw [← s5]; exact h505_0)
  rw [← s0, ← s1, ← s2, ← s3] at hc
  obtain ⟨e, he, hrow⟩ := Rel.row hc 1
  simp only [Matrix.cons_val_one, Matrix.head_cons] at hrow
  rw [e_506_0]
  exact row_mul_qqQ hgq he hrow (by rw [← e_415_0]; exact h416_0) h417_0


set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- Column `5` of the fresh noise times the digits of `Q_q`. -/
theorem freshCol5 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : W_Stage_online_eval.Params}
    {w_358_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_359_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_360_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_361_0 : Fin 4 → ExactMatrix (R i0) 8 1 6}
    {w_393_0 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_1 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_2 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_3 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_4 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_393_5 : Fin 48 → ExactMatrix Q0 8 1 6}
    {w_415_0 : ExactMatrix (R i0) 8 1 1}
    {w_416_0 : ExactMatrix (R i0) 8 6 1}
    {w_417_0 : ExactMatrix (R i0) 8 6 1}
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
    {w_527_0 : ExactMatrix (R i0) 8 1 6}
    {w_528_0 : ExactMatrix (R i0) 8 1 1}
    (e_415_0 : w_415_0 = (matrixPolynomial [((QqZ (i0 : Int)))] : ExactMatrix (R i0) 8 1 1))
    (e_510_0 : w_510_0 = matrixAdd w_508_0 w_509_0)
    (e_513_0 : w_513_0 = matrixAdd w_511_0 w_512_0)
    (e_516_0 : w_516_0 = matrixAdd w_514_0 w_515_0)
    (e_519_0 : w_519_0 = matrixAdd w_517_0 w_518_0)
    (e_522_0 : w_522_0 = matrixAdd w_520_0 w_521_0)
    (e_525_0 : w_525_0 = matrixAdd w_523_0 w_524_0)
    (e_528_0 : w_528_0 = matrixMul w_527_0 w_417_0)
    (h507_5 : (∀ i : Fin 8, W_Stage_online_eval.parallel_parallel_generatedRoot_290_509 Backend.backend model params i0 i ((w_393_0 ⟨i.val + 40, by omega⟩), ((w_393_1 ⟨i.val + 40, by omega⟩), ((w_393_2 ⟨i.val + 40, by omega⟩), ((w_393_3 ⟨i.val + 40, by omega⟩), ((w_393_4 ⟨i.val + 40, by omega⟩), ((w_393_5 ⟨i.val + 40, by omega⟩), ())))))) ((w_507_0 i), ((w_507_1 i), ((w_507_2 i), ((w_507_3 i), ((w_507_4 i), ((w_507_5 i), ()))))))))
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
    (h526_5 : MxxIR.IterRuns (fun (i : Nat) (current next : ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × ExactMatrix Q0 8 1 6 × Unit) => W_Stage_online_eval.sequential_parallel_generatedRoot_290_528 Backend.backend model params i0 i (current.1, (current.2.1, (current.2.2.1, (current.2.2.2.1, (current.2.2.2.2.1, (current.2.2.2.2.2.1, (w_507_0, (w_507_1, (w_507_2, (w_507_3, (w_507_4, (w_507_5, ())))))))))))) next) (Int.toNat (6)) (w_510_0, (w_513_0, (w_516_0, (w_519_0, (w_522_0, (w_525_0, ())))))) (w_526_0, (w_526_1, (w_526_2, (w_526_3, (w_526_4, (w_526_5, ())))))))
    (h527_0 : blockModSwitchRuns [q0, q1, q2, q3] (1) w_526_5 w_527_0)
    (h416_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_415_0 w_416_0)
    (h417_0 : compactCenteredRebaseRuns 16384 w_416_0 w_417_0)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) {ζ : Fin 48 → Nat} {Bs : Nat}
    (hsamp : ∀ o : Fin 48, Rel sm ![![w_393_0 o, w_393_1 o], ![w_393_2 o, w_393_3 o]]
      (fun j ↦ ∑ i : Fin 4, scaleUpMatrix Q0 (qd i0)
        (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) (Xe (2 * ζ o))) •
          (reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
              ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i)))
      ![w_393_4 o, w_393_5 o] Bs)
    {T : Fin 4 → ErrorPoly N} {Δ' : ErrorPoly N} {gq : ExactMatrix (R i0) N 1 6}
    (hgq : gadgetMatrixRuns Backend.backend 32768 6 gq) {inverse : Nat}
    (hinverse : (inverse : ExactPoly (R i0) N) * ((4 : Nat) : ExactPoly (R i0) N) = 1) {BS : Nat}
    (hswitch : ∀ i : Fin 4, ∃ e : Fin 2 → ErrorMatrix N 1 6, (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly (R i0) N (substitution (sigma i).toNat (sigma_odd i) sp) •
          ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 0 j i +
            ![![w_358_0, w_359_0], ![w_360_0, w_361_0]] 1 j i =
        (lutCoefficient (R i0) inverse (fun k ↦ reducePoly (R i0) N (T k)) (sigma i).toNat *
          reducePoly (R i0) N (Δ' * unitPair sp j)) • gq + reduceMatrix (R i0) N 1 6 (e j)) :
    ∃ e : ErrorMatrix N 1 1,
      CoeffBound e (6 * N * ((8 * Bs + (N + 2) * ((qd i0) / 2)) / (qd i0) + 32 * (N * 1 * BS)) * 16384) ∧
      w_528_0 = maskPart (reducePoly (R i0) N sm) (fun k ↦
          blockModSwitch [q0, q1, q2, q3] 1 (![![w_526_0, w_526_1], ![w_526_2, w_526_3]] k 1) * w_417_0) -
        reducePoly (R i0) N (Δ' * (∑ i : Fin 8, Xe (n := N) i * sampled T (ζ ⟨i.val + 40, by omega⟩)) *
          unitPair sp 1) • (matrixPolynomial [(QqZ (i0 : Int))] : ExactMatrix (R i0) N 1 1) +
        reduceMatrix (R i0) N 1 1 e := by
  subst e_510_0 e_513_0 e_516_0 e_519_0 e_522_0 e_525_0
  have hs := rootSum526 h508_0 h509_0 h511_0 h512_0 h514_0 h515_0 h517_0 h518_0 h520_0 h521_0 h523_0 h524_0 h526_5
  simp only [Prod.mk.injEq] at hs
  obtain ⟨s0, s1, s2, s3, s4, s5, -⟩ := hs
  have hc := freshColumn5 (model := model) (hplace := h507_5) hsm (ζ := ζ) hsamp hinverse hswitch
    (by rw [← s5]; exact h527_0)
  rw [← s0, ← s1, ← s2, ← s3] at hc
  obtain ⟨e, he, hrow⟩ := Rel.row hc 1
  simp only [Matrix.cons_val_one, Matrix.head_cons] at hrow
  rw [e_528_0]
  exact row_mul_qqQ hgq he hrow (by rw [← e_415_0]; exact h416_0) h417_0


end RefreshTopProof.Part_I
