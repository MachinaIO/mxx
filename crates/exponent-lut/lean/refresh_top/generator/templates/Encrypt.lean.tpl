import TopLemmas

/-!
# The encryption stage

The encryptions of the children of every part: the state input of each PRF at its top modulus.
-/

namespace RefreshTopProof.Part_@P@

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof

set_option maxHeartbeats 16000000 in
set_option maxRecDepth 65536 in
theorem mask_state {model : HashModel} {params} {inputs} {outputs}
    (h : @E@.generatedRoot Backend.backend model params inputs outputs)
    {sm sp : ErrorPoly N} (hsm : polyNorm sm ≤ 1) (hsp : polyNorm sp ≤ 1)
    (hmask : @INPUT(encrypt, w_4_0)@ = fun _ _ ↦ reducePoly Q0 N sm)
    (hpay : @INPUT(encrypt, w_11_0)@ = fun _ _ ↦ reducePoly Q0 N sp)
    {g' : ExactMatrix Q6 N 1 12} {g : ExactMatrix Q7 N 1 12}
    (hg' : gadgetMatrixRuns Backend.backend 32768 12 g') (hg : centeredRebaseRuns g' g) :
    Rel sm ![![@INPUT(encrypt, w_337_0)@, @INPUT(encrypt, w_338_0)@], ![@INPUT(encrypt, w_339_0)@, @INPUT(encrypt, w_340_0)@]]
      (fun j ↦ reducePoly Q7 N (((p6 : Nat) : Int) * message (n := N) @INPUT(encrypt, w_351_0)@ @INPUT(encrypt, w_353_0)@ *
        unitPair sp j) • g)
      ![@OUTPUT(encrypt, w_359_0)@, @OUTPUT(encrypt, w_368_0)@] 26 := by
@DESTRUCT(encrypt, generatedRoot, h)@
  rw [gadgetMatrixRuns_unique h347_0 hg'] at h348_0
  have hgg := centeredRebaseRuns_unique h348_0 hg
  obtain ⟨e0, he0, hb0⟩ := gaussian_bound h358_0
  obtain ⟨e1, he1, hb1⟩ := gaussian_bound h367_0
  have hm4 : w_4_0 = fun _ _ ↦ reducePoly Q0 N sm := hmask
  have hp11 : w_11_0 = fun _ _ ↦ reducePoly Q0 N sp := hpay
  have hrm : w_341_0 = fun _ _ ↦ reducePoly Q7 N sm := rebase_small (q := Q0) (by norm_num) hsm (by rw [← hm4]; exact h341_0)
  have hrp : w_346_0 = fun _ _ ↦ reducePoly Q7 N sp := rebase_small (q := Q0) (by norm_num) hsp (by rw [← hp11]; exact h346_0)
  have hr1 : w_343_0 = 1 := rebase_one (q := Q0) (by norm_num) h343_0
  have hr29 : w_363_0 = 1 := rebase_one (q := Q0) (by norm_num) h363_0
  subst hout
  refine ⟨![e0, e1], fun j ↦ by fin_cases j <;> assumption, fun j ↦ ?_⟩
  fin_cases j
  · show w_359_0 = _
    simp only [w_359_0, w_357_0, w_345_0, w_342_0, w_344_0, w_356_0, w_349_0, w_355_0, w_350_0,
      w_354_0, w_352_0, w_337_0, w_339_0, w_351_0, w_353_0, hrm, hr1, hrp, he0, hgg,
      multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring
  · show w_368_0 = _
    simp only [w_368_0, w_366_0, w_362_0, w_360_0, w_361_0, w_365_0, w_364_0, w_355_0, w_350_0,
      w_354_0, w_352_0, w_338_0, w_340_0, w_351_0, w_353_0, hrm, hr1, hr29, he1, hgg,
      multiplyMonomial_liftInteger]
    simp only [matrixMulScalarRight_smul, matrixMulScalarLeft_eq]
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    simp [matrixAdd, matrixSub, unitPair, reduceMatrix, matrixPolynomial_single, map_ofNat]
    try ring

end RefreshTopProof.Part_@P@
