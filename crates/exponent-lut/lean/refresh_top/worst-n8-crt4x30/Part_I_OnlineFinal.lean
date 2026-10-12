import Part_I_Local
import TopInst
import TopColumn
import TopTarget
import TopResidue
import TopChain
import Stage_online_eval

/-! The online root's output of part `I` (written by `generator/gen_online_final.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof Mxx.ExponentLut.Refresh

set_option maxHeartbeats 64000000 in
set_option maxRecDepth 65536 in
/-- The online root's output: modulo `q₀`, the refresh's value `y`. -/
theorem onlineFinal {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_online_eval.Params}
    {w_84_0 : ExactMatrix (R i0) 8 1 16}
    {w_85_0 : ExactMatrix (R i0) 8 16 1}
    {w_334_0 : ExactMatrix (R i0) 8 1 6}
    {w_584_0 : ExactMatrix (R i0) 8 16 1}
    {w_638_0 : ExactMatrix (R i0) 8 16 1}
    {w_692_0 : ExactMatrix (R i0) 8 16 1}
    {w_746_0 : ExactMatrix (R i0) 8 16 1}
    {w_800_0 : ExactMatrix (R i0) 8 16 1}
    {w_86_0 : ExactMatrix (R i0) 8 1 1}
    {w_333_0 : ExactMatrix (R i0) 8 1 6}
    {w_335_0 : ExactMatrix (R i0) 8 1 12}
    {w_336_0 : ExactMatrix (R i0) 8 1 1}
    {w_338_0 : ExactMatrix Q0 8 2 6}
    {w_339_0 : ExactMatrix (R i0) 8 2 6}
    {w_340_0 : ExactMatrix (R i0) 8 2 6}
    {w_341_0 : ExactMatrix (R i0) 8 12 6}
    {w_342_0 : ExactMatrix (R i0) 8 12 6}
    {w_343_0 : ExactMatrix (R i0) 8 1 6}
    {w_10345_0 : ExactMatrix (R i0) 8 1 4}
    {w_346_0 : ExactMatrix (qc i0) 8 1 6}
    {w_347_0 : ExactMatrix (qc i0) 8 1 1}
    {w_348_0 : ExactMatrix (qc i0) 8 1 6}
    {w_349_0 : ExactMatrix (qc i0) 8 1 1}
    {w_350_0 : ExactMatrix (qc i0) 8 1 6}
    {w_351_0 : ExactMatrix (Mx i0) 8 1 6}
    {w_352_0 : ExactMatrix (Mx i0) 8 1 1}
    {w_353_0 : ExactMatrix (Mx i0) 8 1 6}
    {w_354_0 : ExactMatrix (Mx i0) 8 4 6}
    {w_355_0 : ExactMatrix (R i0) 8 4 6}
    {w_356_0 : ExactMatrix (R i0) 8 1 6}
    {w_357_0 : ExactMatrix (R i0) 8 1 6}
    {w_529_0 : ExactMatrix (R i0) 8 1 6}
    {w_530_0 : ExactMatrix (R i0) 8 1 6}
    {w_531_0 : ExactMatrix (R i0) 8 1 1}
    {w_579_5 : ExactMatrix (R i0) 8 1 1}
    {w_580_0 : ExactMatrix (R i0) 8 1 1}
    {w_581_0 : ExactMatrix (R i0) 8 1 1}
    {w_582_0 : ExactMatrix (R i0) 8 1 1}
    {w_583_0 : ExactMatrix (R i0) 8 1 1}
    {w_585_0 : ExactMatrix (R i0) 8 1 1}
    {w_586_0 : ExactMatrix (R i0) 8 1 1}
    {w_634_5 : ExactMatrix (R i0) 8 1 1}
    {w_635_0 : ExactMatrix (R i0) 8 1 1}
    {w_636_0 : ExactMatrix (R i0) 8 1 1}
    {w_637_0 : ExactMatrix (R i0) 8 1 1}
    {w_639_0 : ExactMatrix (R i0) 8 1 1}
    {w_640_0 : ExactMatrix (R i0) 8 1 1}
    {w_688_5 : ExactMatrix (R i0) 8 1 1}
    {w_689_0 : ExactMatrix (R i0) 8 1 1}
    {w_690_0 : ExactMatrix (R i0) 8 1 1}
    {w_691_0 : ExactMatrix (R i0) 8 1 1}
    {w_693_0 : ExactMatrix (R i0) 8 1 1}
    {w_694_0 : ExactMatrix (R i0) 8 1 1}
    {w_742_5 : ExactMatrix (R i0) 8 1 1}
    {w_743_0 : ExactMatrix (R i0) 8 1 1}
    {w_744_0 : ExactMatrix (R i0) 8 1 1}
    {w_745_0 : ExactMatrix (R i0) 8 1 1}
    {w_747_0 : ExactMatrix (R i0) 8 1 1}
    {w_748_0 : ExactMatrix (R i0) 8 1 1}
    {w_796_5 : ExactMatrix (R i0) 8 1 1}
    {w_797_0 : ExactMatrix (R i0) 8 1 1}
    {w_798_0 : ExactMatrix (R i0) 8 1 1}
    {w_799_0 : ExactMatrix (R i0) 8 1 1}
    {w_801_0 : ExactMatrix (R i0) 8 1 1}
    {w_802_0 : ExactMatrix (R i0) 8 1 1}
    {w_850_5 : ExactMatrix (R i0) 8 1 1}
    {w_851_0 : ExactMatrix (R i0) 8 1 1}
    {w_852_0 : ExactMatrix (R i0) 8 1 1}
    {w_853_0 : ExactMatrix (R i0) 8 1 1}
    {w_854_0 : ExactMatrix (R i0) 8 1 6}
    {w_854_concat_1 : ExactMatrix (R i0) 8 1 2}
    {w_854_concat_2 : ExactMatrix (R i0) 8 1 3}
    {w_854_concat_3 : ExactMatrix (R i0) 8 1 4}
    {w_854_concat_4 : ExactMatrix (R i0) 8 1 5}
    {w_855_0 : ExactMatrix (qc i0) 8 1 6}
    (e_86_0 : w_86_0 = matrixMul w_84_0 w_85_0)
    (e_336_0 : w_336_0 = (matrixPolynomial [((QqZ (i0 : Int)))] : ExactMatrix (R i0) 8 1 1))
    (e_340_0 : w_340_0 = matrixMulScalarLeft w_336_0 w_339_0)
    (e_343_0 : w_343_0 = matrixMul w_335_0 w_342_0)
    (e_347_0 : w_347_0 = (matrixPolynomial [((k3Z (i0 : Int)))] : ExactMatrix (qc i0) 8 1 1))
    (e_348_0 : w_348_0 = matrixMulScalarRight w_346_0 w_347_0)
    (e_349_0 : w_349_0 = (matrixPolynomial [((k1Z (i0 : Int)))] : ExactMatrix (qc i0) 8 1 1))
    (e_350_0 : w_350_0 = matrixMulScalarRight w_348_0 w_349_0)
    (e_352_0 : w_352_0 = (matrixPolynomial [((cpZ (i0 : Int)))] : ExactMatrix (Mx i0) 8 1 1))
    (e_353_0 : w_353_0 = matrixMulScalarRight w_351_0 w_352_0)
    (e_356_0 : w_356_0 = matrixMul w_10345_0 w_355_0)
    (e_357_0 : w_357_0 = matrixSub w_343_0 w_356_0)
    (e_530_0 : w_530_0 = matrixAdd w_357_0 w_529_0)
    (e_580_0 : w_580_0 = matrixAdd w_531_0 w_579_5)
    (e_581_0 : w_581_0 = matrixSub w_86_0 w_580_0)
    (e_582_0 : w_582_0 = (matrixPolynomial [((halfZ (i0 : Int))), ((halfZ (i0 : Int))), ((halfZ (i0 : Int))), ((halfZ (i0 : Int))), ((halfZ (i0 : Int))), ((halfZ (i0 : Int))), ((halfZ (i0 : Int))), ((halfZ (i0 : Int)))] : ExactMatrix (R i0) 8 1 1))
    (e_583_0 : w_583_0 = matrixSub w_581_0 w_582_0)
    (e_585_0 : w_585_0 = matrixMul w_84_0 w_584_0)
    (e_635_0 : w_635_0 = matrixAdd w_586_0 w_634_5)
    (e_636_0 : w_636_0 = matrixSub w_585_0 w_635_0)
    (e_637_0 : w_637_0 = matrixSub w_636_0 w_582_0)
    (e_639_0 : w_639_0 = matrixMul w_84_0 w_638_0)
    (e_689_0 : w_689_0 = matrixAdd w_640_0 w_688_5)
    (e_690_0 : w_690_0 = matrixSub w_639_0 w_689_0)
    (e_691_0 : w_691_0 = matrixSub w_690_0 w_582_0)
    (e_693_0 : w_693_0 = matrixMul w_84_0 w_692_0)
    (e_743_0 : w_743_0 = matrixAdd w_694_0 w_742_5)
    (e_744_0 : w_744_0 = matrixSub w_693_0 w_743_0)
    (e_745_0 : w_745_0 = matrixSub w_744_0 w_582_0)
    (e_747_0 : w_747_0 = matrixMul w_84_0 w_746_0)
    (e_797_0 : w_797_0 = matrixAdd w_748_0 w_796_5)
    (e_798_0 : w_798_0 = matrixSub w_747_0 w_797_0)
    (e_799_0 : w_799_0 = matrixSub w_798_0 w_582_0)
    (e_801_0 : w_801_0 = matrixMul w_84_0 w_800_0)
    (e_851_0 : w_851_0 = matrixAdd w_802_0 w_850_5)
    (e_852_0 : w_852_0 = matrixSub w_801_0 w_851_0)
    (e_853_0 : w_853_0 = matrixSub w_852_0 w_582_0)
    (h335_0 : concatColumns w_333_0 w_334_0 w_335_0)
    (h339_0 : modulusReduceRuns w_338_0 w_339_0)
    (h341_0 : gadgetDecomposeRuns Backend.backend 32768 6 w_340_0 w_341_0)
    (h342_0 : compactCenteredRebaseRuns 16384 w_341_0 w_342_0)
    (h351_0 : centeredRebaseRuns w_350_0 w_351_0)
    (h354_0 : gadgetDecomposeRuns Backend.backend 32768 4 w_353_0 w_354_0)
    (h355_0 : compactCenteredRebaseRuns 16384 w_354_0 w_355_0)
    (h637_0 : concatColumns w_583_0 w_637_0 w_854_concat_1)
    (h691_0 : concatColumns w_854_concat_1 w_691_0 w_854_concat_2)
    (h745_0 : concatColumns w_854_concat_2 w_745_0 w_854_concat_3)
    (h799_0 : concatColumns w_854_concat_3 w_799_0 w_854_concat_4)
    (h854_0 : concatColumns w_854_concat_4 w_853_0 w_854_0)
    (h855_0 : modulusSwitchRuns w_854_0 w_855_0)
    (h531_0 : sliceMatrix w_530_0 0 1 0 1 w_531_0)
    (h586_0 : sliceMatrix w_530_0 0 1 1 2 w_586_0)
    (h640_0 : sliceMatrix w_530_0 0 1 2 3 w_640_0)
    (h694_0 : sliceMatrix w_530_0 0 1 3 4 w_694_0)
    (h748_0 : sliceMatrix w_530_0 0 1 4 5 w_748_0)
    (h802_0 : sliceMatrix w_530_0 0 1 5 6 w_802_0)
    {sm : ErrorPoly N} {β : ExactPoly (R i0) N} {x : ErrorPoly N}
    {Kg K₁ Kw : Fin 2 → ExactMatrix (R i0) N 1 6} {Kx : Fin 2 → ExactMatrix (R i0) N 1 4}
    {Bh : Fin 2 → ExactMatrix (R i0) N 1 16} {e' : ExactMatrix (R i0) N 1 6}
    {GQ1_6 : ExactMatrix (R i0) N 1 6} (hGQ1_6 : gadgetMatrixRuns Backend.backend 32768 6 GQ1_6)
    {GM01_4 : ExactMatrix (Mx i0) N 1 4} {RQ1_4 : ExactMatrix (R i0) N 1 4}
    (hGM01_4 : gadgetMatrixRuns Backend.backend 32768 4 GM01_4) (hRQ1_4 : centeredRebaseRuns GM01_4 RQ1_4)
    {egI e1I ewI : ErrorMatrix N 1 6} {exI : ErrorMatrix N 1 4} {ebI : ErrorMatrix N 1 16}
    {Beg Be1 Bew Bex Beb Bpre BM : Nat}
    (heg : CoeffBound egI Beg) (he1 : CoeffBound e1I Be1) (hew : CoeffBound ewI Bew)
    (hex : CoeffBound exI Bex) (heb : CoeffBound ebI Beb)
    (hg : w_333_0 = maskPart (reducePoly (R i0) N sm) Kg - β • GQ1_6 + reduceMatrix (R i0) N 1 6 egI)
    (hc₁ : w_334_0 = maskPart (reducePoly (R i0) N sm) K₁ - GQ1_6 + reduceMatrix (R i0) N 1 6 e1I)
    (hcx : w_10345_0 = maskPart (reducePoly (R i0) N sm) Kx - reducePoly (R i0) N ((((pp i0) : Nat) : Int) * x) • RQ1_4 +
      reduceMatrix (R i0) N 1 4 exI)
    (hfw : w_529_0 = maskPart (reducePoly (R i0) N sm) Kw - (((QqZ (i0 : Int)) : Int) : ExactPoly (R i0) N) • e' + reduceMatrix (R i0) N 1 6 ewI)
    (hb : w_84_0 = maskPart (reducePoly (R i0) N sm) Bh + reduceMatrix (R i0) N 1 16 ebI)
    {KI0 : ErrorMatrix N 16 1} (hK0 : w_85_0 = reduceMatrix (R i0) N 16 1 KI0) (hKb0 : CoeffBound KI0 Bpre)
    {Ksm0 : Fin 2 → ExactMatrix (R i0) N 1 1} {V0 M0 : ErrorPoly N} {esm0 : ErrorMatrix N 1 1} {Besm0 : Nat}
    (hpre0 : ∀ i, Bh i * w_85_0 = (Kg i * topRows w_342_0 + K₁ i * bottomRows w_342_0 - Kx i * w_355_0 + Kw i) *
      unitCol (q := (R i0)) 0 + Ksm0 i)
    (hsm0 : w_579_5 = maskPart (reducePoly (R i0) N sm) Ksm0 - reducePoly (R i0) N V0 • (1 : ExactMatrix (R i0) N 1 1) +
      reduceMatrix (R i0) N 1 1 esm0) (hesm0 : CoeffBound esm0 Besm0)
    (hV0 : reducePoly (R i0) N V0 - w_582_0 0 0 = reducePoly (R i0) N M0)
    (hM0 : CoeffBound (fun _ _ ↦ M0 : ErrorMatrix N 1 1) BM)
    (hnoise0 : 2 * (BM + 16 * N * Beb * Bpre + (6 * N * Beg * 16384 + 6 * N * Be1 * 16384 +
      4 * N * Bex * 16384 + Bew) + Besm0) < (Qq i0))
    {KI1 : ErrorMatrix N 16 1} (hK1 : w_584_0 = reduceMatrix (R i0) N 16 1 KI1) (hKb1 : CoeffBound KI1 Bpre)
    {Ksm1 : Fin 2 → ExactMatrix (R i0) N 1 1} {V1 M1 : ErrorPoly N} {esm1 : ErrorMatrix N 1 1} {Besm1 : Nat}
    (hpre1 : ∀ i, Bh i * w_584_0 = (Kg i * topRows w_342_0 + K₁ i * bottomRows w_342_0 - Kx i * w_355_0 + Kw i) *
      unitCol (q := (R i0)) 1 + Ksm1 i)
    (hsm1 : w_634_5 = maskPart (reducePoly (R i0) N sm) Ksm1 - reducePoly (R i0) N V1 • (1 : ExactMatrix (R i0) N 1 1) +
      reduceMatrix (R i0) N 1 1 esm1) (hesm1 : CoeffBound esm1 Besm1)
    (hV1 : reducePoly (R i0) N V1 - w_582_0 0 0 = reducePoly (R i0) N M1)
    (hM1 : CoeffBound (fun _ _ ↦ M1 : ErrorMatrix N 1 1) BM)
    (hnoise1 : 2 * (BM + 16 * N * Beb * Bpre + (6 * N * Beg * 16384 + 6 * N * Be1 * 16384 +
      4 * N * Bex * 16384 + Bew) + Besm1) < (Qq i0))
    {KI2 : ErrorMatrix N 16 1} (hK2 : w_638_0 = reduceMatrix (R i0) N 16 1 KI2) (hKb2 : CoeffBound KI2 Bpre)
    {Ksm2 : Fin 2 → ExactMatrix (R i0) N 1 1} {V2 M2 : ErrorPoly N} {esm2 : ErrorMatrix N 1 1} {Besm2 : Nat}
    (hpre2 : ∀ i, Bh i * w_638_0 = (Kg i * topRows w_342_0 + K₁ i * bottomRows w_342_0 - Kx i * w_355_0 + Kw i) *
      unitCol (q := (R i0)) 2 + Ksm2 i)
    (hsm2 : w_688_5 = maskPart (reducePoly (R i0) N sm) Ksm2 - reducePoly (R i0) N V2 • (1 : ExactMatrix (R i0) N 1 1) +
      reduceMatrix (R i0) N 1 1 esm2) (hesm2 : CoeffBound esm2 Besm2)
    (hV2 : reducePoly (R i0) N V2 - w_582_0 0 0 = reducePoly (R i0) N M2)
    (hM2 : CoeffBound (fun _ _ ↦ M2 : ErrorMatrix N 1 1) BM)
    (hnoise2 : 2 * (BM + 16 * N * Beb * Bpre + (6 * N * Beg * 16384 + 6 * N * Be1 * 16384 +
      4 * N * Bex * 16384 + Bew) + Besm2) < (Qq i0))
    {KI3 : ErrorMatrix N 16 1} (hK3 : w_692_0 = reduceMatrix (R i0) N 16 1 KI3) (hKb3 : CoeffBound KI3 Bpre)
    {Ksm3 : Fin 2 → ExactMatrix (R i0) N 1 1} {V3 M3 : ErrorPoly N} {esm3 : ErrorMatrix N 1 1} {Besm3 : Nat}
    (hpre3 : ∀ i, Bh i * w_692_0 = (Kg i * topRows w_342_0 + K₁ i * bottomRows w_342_0 - Kx i * w_355_0 + Kw i) *
      unitCol (q := (R i0)) 3 + Ksm3 i)
    (hsm3 : w_742_5 = maskPart (reducePoly (R i0) N sm) Ksm3 - reducePoly (R i0) N V3 • (1 : ExactMatrix (R i0) N 1 1) +
      reduceMatrix (R i0) N 1 1 esm3) (hesm3 : CoeffBound esm3 Besm3)
    (hV3 : reducePoly (R i0) N V3 - w_582_0 0 0 = reducePoly (R i0) N M3)
    (hM3 : CoeffBound (fun _ _ ↦ M3 : ErrorMatrix N 1 1) BM)
    (hnoise3 : 2 * (BM + 16 * N * Beb * Bpre + (6 * N * Beg * 16384 + 6 * N * Be1 * 16384 +
      4 * N * Bex * 16384 + Bew) + Besm3) < (Qq i0))
    {KI4 : ErrorMatrix N 16 1} (hK4 : w_746_0 = reduceMatrix (R i0) N 16 1 KI4) (hKb4 : CoeffBound KI4 Bpre)
    {Ksm4 : Fin 2 → ExactMatrix (R i0) N 1 1} {V4 M4 : ErrorPoly N} {esm4 : ErrorMatrix N 1 1} {Besm4 : Nat}
    (hpre4 : ∀ i, Bh i * w_746_0 = (Kg i * topRows w_342_0 + K₁ i * bottomRows w_342_0 - Kx i * w_355_0 + Kw i) *
      unitCol (q := (R i0)) 4 + Ksm4 i)
    (hsm4 : w_796_5 = maskPart (reducePoly (R i0) N sm) Ksm4 - reducePoly (R i0) N V4 • (1 : ExactMatrix (R i0) N 1 1) +
      reduceMatrix (R i0) N 1 1 esm4) (hesm4 : CoeffBound esm4 Besm4)
    (hV4 : reducePoly (R i0) N V4 - w_582_0 0 0 = reducePoly (R i0) N M4)
    (hM4 : CoeffBound (fun _ _ ↦ M4 : ErrorMatrix N 1 1) BM)
    (hnoise4 : 2 * (BM + 16 * N * Beb * Bpre + (6 * N * Beg * 16384 + 6 * N * Be1 * 16384 +
      4 * N * Bex * 16384 + Bew) + Besm4) < (Qq i0))
    {KI5 : ErrorMatrix N 16 1} (hK5 : w_800_0 = reduceMatrix (R i0) N 16 1 KI5) (hKb5 : CoeffBound KI5 Bpre)
    {Ksm5 : Fin 2 → ExactMatrix (R i0) N 1 1} {V5 M5 : ErrorPoly N} {esm5 : ErrorMatrix N 1 1} {Besm5 : Nat}
    (hpre5 : ∀ i, Bh i * w_800_0 = (Kg i * topRows w_342_0 + K₁ i * bottomRows w_342_0 - Kx i * w_355_0 + Kw i) *
      unitCol (q := (R i0)) 5 + Ksm5 i)
    (hsm5 : w_850_5 = maskPart (reducePoly (R i0) N sm) Ksm5 - reducePoly (R i0) N V5 • (1 : ExactMatrix (R i0) N 1 1) +
      reduceMatrix (R i0) N 1 1 esm5) (hesm5 : CoeffBound esm5 Besm5)
    (hV5 : reducePoly (R i0) N V5 - w_582_0 0 0 = reducePoly (R i0) N M5)
    (hM5 : CoeffBound (fun _ _ ↦ M5 : ErrorMatrix N 1 1) BM)
    (hnoise5 : 2 * (BM + 16 * N * Beb * Bpre + (6 * N * Beg * 16384 + 6 * N * Be1 * 16384 +
      4 * N * Bex * 16384 + Bew) + Besm5) < (Qq i0)) :
    ∀ (k : Fin 6) (i : Fin N), (w_855_0 0 k).coeff i =
      ((((β • rowOf w_339_0 0 + rowOf w_339_0 1 - actionValueC (actionConst' (cpZ (i0 : Int)) (kdiv i0)) x w_350_0 + e' : ExactMatrix (R i0) N 1 6) 0 k).coeff i).val : ZMod (qc i0)) := by
  obtain ⟨L, hL, hLx, hLb, hLd⟩ := layoutR i0 hi0
  obtain ⟨LX, hLX, hLXx, -, -⟩ := layoutMx i0 hi0
  obtain ⟨DI, hDI, hDIb⟩ := h342_0.2.2.2
  obtain ⟨YI, hYI, hYIb⟩ := h355_0.2.2.2
  have hcol0 := column_valueR (Qq := (QqZ (i0 : Int))) (c' := (cpZ (i0 : Int))) (k := actionConst' (cpZ (i0 : Int)) (kdiv i0)) L hL hLx (by inst_num) (by rw [hLb]; norm_num) (by rw [hLd]; norm_num) (by rw [hLd])
    (p := (pp i0)) (Mx := (Mx i0)) (R_eq_pMx i0 hi0) (by inst_num) LX hLX hLXx (action_cert i0 hi0)
    (G := GQ1_6) (H := w_582_0) h335_0 (z := w_340_0) (A' := w_339_0) (by rw [e_340_0, e_336_0]) h341_0 h342_0 hGQ1_6
    h351_0 (by rw [e_353_0, e_352_0]) h354_0 h355_0 hGM01_4 hRQ1_4 (comb := w_530_0)
    (by rw [e_530_0, e_357_0, e_343_0, e_356_0]) (by decide : ((0 : Int)).toNat < 6) h531_0 hg hc₁ hcx hfw hb
    (Kpre := w_85_0) hpre0 hsm0 (out := w_583_0) (by rw [e_583_0, e_581_0, e_580_0, e_86_0]; try rfl)
  have hSH0 : reducePoly (R i0) N V0 • (1 : ExactMatrix (R i0) N 1 1) - w_582_0 =
      reduceMatrix (R i0) N 1 1 (fun _ _ ↦ M0) := by
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.one_apply_eq, smul_eq_mul, mul_one, reduceMatrix]
    exact hV0
  rw [hSH0, hK0, hDI, hYI] at hcol0
  obtain ⟨ν0, hν0, hc0⟩ := column_noiseQ (Qq := (QqZ (i0 : Int))) (y := β • rowOf w_339_0 0 + rowOf w_339_0 1 - actionValueC (actionConst' (cpZ (i0 : Int)) (kdiv i0)) x w_350_0 + e')
    0 hM0 heb hKb0 heg he1 hew (X0I := topRows DI) (X1I := bottomRows DI)
    (fun r c i ↦ hDIb _ c i) (fun r c i ↦ hDIb _ c i) hex hYIb hesm0 (out := w_583_0) (by rw [hcol0]; try rfl)
  have hcol1 := column_valueR (Qq := (QqZ (i0 : Int))) (c' := (cpZ (i0 : Int))) (k := actionConst' (cpZ (i0 : Int)) (kdiv i0)) L hL hLx (by inst_num) (by rw [hLb]; norm_num) (by rw [hLd]; norm_num) (by rw [hLd])
    (p := (pp i0)) (Mx := (Mx i0)) (R_eq_pMx i0 hi0) (by inst_num) LX hLX hLXx (action_cert i0 hi0)
    (G := GQ1_6) (H := w_582_0) h335_0 (z := w_340_0) (A' := w_339_0) (by rw [e_340_0, e_336_0]) h341_0 h342_0 hGQ1_6
    h351_0 (by rw [e_353_0, e_352_0]) h354_0 h355_0 hGM01_4 hRQ1_4 (comb := w_530_0)
    (by rw [e_530_0, e_357_0, e_343_0, e_356_0]) (by decide : ((1 : Int)).toNat < 6) h586_0 hg hc₁ hcx hfw hb
    (Kpre := w_584_0) hpre1 hsm1 (out := w_637_0) (by rw [e_637_0, e_636_0, e_635_0, e_585_0]; try rfl)
  have hSH1 : reducePoly (R i0) N V1 • (1 : ExactMatrix (R i0) N 1 1) - w_582_0 =
      reduceMatrix (R i0) N 1 1 (fun _ _ ↦ M1) := by
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.one_apply_eq, smul_eq_mul, mul_one, reduceMatrix]
    exact hV1
  rw [hSH1, hK1, hDI, hYI] at hcol1
  obtain ⟨ν1, hν1, hc1⟩ := column_noiseQ (Qq := (QqZ (i0 : Int))) (y := β • rowOf w_339_0 0 + rowOf w_339_0 1 - actionValueC (actionConst' (cpZ (i0 : Int)) (kdiv i0)) x w_350_0 + e')
    1 hM1 heb hKb1 heg he1 hew (X0I := topRows DI) (X1I := bottomRows DI)
    (fun r c i ↦ hDIb _ c i) (fun r c i ↦ hDIb _ c i) hex hYIb hesm1 (out := w_637_0) (by rw [hcol1]; try rfl)
  have hcol2 := column_valueR (Qq := (QqZ (i0 : Int))) (c' := (cpZ (i0 : Int))) (k := actionConst' (cpZ (i0 : Int)) (kdiv i0)) L hL hLx (by inst_num) (by rw [hLb]; norm_num) (by rw [hLd]; norm_num) (by rw [hLd])
    (p := (pp i0)) (Mx := (Mx i0)) (R_eq_pMx i0 hi0) (by inst_num) LX hLX hLXx (action_cert i0 hi0)
    (G := GQ1_6) (H := w_582_0) h335_0 (z := w_340_0) (A' := w_339_0) (by rw [e_340_0, e_336_0]) h341_0 h342_0 hGQ1_6
    h351_0 (by rw [e_353_0, e_352_0]) h354_0 h355_0 hGM01_4 hRQ1_4 (comb := w_530_0)
    (by rw [e_530_0, e_357_0, e_343_0, e_356_0]) (by decide : ((2 : Int)).toNat < 6) h640_0 hg hc₁ hcx hfw hb
    (Kpre := w_638_0) hpre2 hsm2 (out := w_691_0) (by rw [e_691_0, e_690_0, e_689_0, e_639_0]; try rfl)
  have hSH2 : reducePoly (R i0) N V2 • (1 : ExactMatrix (R i0) N 1 1) - w_582_0 =
      reduceMatrix (R i0) N 1 1 (fun _ _ ↦ M2) := by
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.one_apply_eq, smul_eq_mul, mul_one, reduceMatrix]
    exact hV2
  rw [hSH2, hK2, hDI, hYI] at hcol2
  obtain ⟨ν2, hν2, hc2⟩ := column_noiseQ (Qq := (QqZ (i0 : Int))) (y := β • rowOf w_339_0 0 + rowOf w_339_0 1 - actionValueC (actionConst' (cpZ (i0 : Int)) (kdiv i0)) x w_350_0 + e')
    2 hM2 heb hKb2 heg he1 hew (X0I := topRows DI) (X1I := bottomRows DI)
    (fun r c i ↦ hDIb _ c i) (fun r c i ↦ hDIb _ c i) hex hYIb hesm2 (out := w_691_0) (by rw [hcol2]; try rfl)
  have hcol3 := column_valueR (Qq := (QqZ (i0 : Int))) (c' := (cpZ (i0 : Int))) (k := actionConst' (cpZ (i0 : Int)) (kdiv i0)) L hL hLx (by inst_num) (by rw [hLb]; norm_num) (by rw [hLd]; norm_num) (by rw [hLd])
    (p := (pp i0)) (Mx := (Mx i0)) (R_eq_pMx i0 hi0) (by inst_num) LX hLX hLXx (action_cert i0 hi0)
    (G := GQ1_6) (H := w_582_0) h335_0 (z := w_340_0) (A' := w_339_0) (by rw [e_340_0, e_336_0]) h341_0 h342_0 hGQ1_6
    h351_0 (by rw [e_353_0, e_352_0]) h354_0 h355_0 hGM01_4 hRQ1_4 (comb := w_530_0)
    (by rw [e_530_0, e_357_0, e_343_0, e_356_0]) (by decide : ((3 : Int)).toNat < 6) h694_0 hg hc₁ hcx hfw hb
    (Kpre := w_692_0) hpre3 hsm3 (out := w_745_0) (by rw [e_745_0, e_744_0, e_743_0, e_693_0]; try rfl)
  have hSH3 : reducePoly (R i0) N V3 • (1 : ExactMatrix (R i0) N 1 1) - w_582_0 =
      reduceMatrix (R i0) N 1 1 (fun _ _ ↦ M3) := by
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.one_apply_eq, smul_eq_mul, mul_one, reduceMatrix]
    exact hV3
  rw [hSH3, hK3, hDI, hYI] at hcol3
  obtain ⟨ν3, hν3, hc3⟩ := column_noiseQ (Qq := (QqZ (i0 : Int))) (y := β • rowOf w_339_0 0 + rowOf w_339_0 1 - actionValueC (actionConst' (cpZ (i0 : Int)) (kdiv i0)) x w_350_0 + e')
    3 hM3 heb hKb3 heg he1 hew (X0I := topRows DI) (X1I := bottomRows DI)
    (fun r c i ↦ hDIb _ c i) (fun r c i ↦ hDIb _ c i) hex hYIb hesm3 (out := w_745_0) (by rw [hcol3]; try rfl)
  have hcol4 := column_valueR (Qq := (QqZ (i0 : Int))) (c' := (cpZ (i0 : Int))) (k := actionConst' (cpZ (i0 : Int)) (kdiv i0)) L hL hLx (by inst_num) (by rw [hLb]; norm_num) (by rw [hLd]; norm_num) (by rw [hLd])
    (p := (pp i0)) (Mx := (Mx i0)) (R_eq_pMx i0 hi0) (by inst_num) LX hLX hLXx (action_cert i0 hi0)
    (G := GQ1_6) (H := w_582_0) h335_0 (z := w_340_0) (A' := w_339_0) (by rw [e_340_0, e_336_0]) h341_0 h342_0 hGQ1_6
    h351_0 (by rw [e_353_0, e_352_0]) h354_0 h355_0 hGM01_4 hRQ1_4 (comb := w_530_0)
    (by rw [e_530_0, e_357_0, e_343_0, e_356_0]) (by decide : ((4 : Int)).toNat < 6) h748_0 hg hc₁ hcx hfw hb
    (Kpre := w_746_0) hpre4 hsm4 (out := w_799_0) (by rw [e_799_0, e_798_0, e_797_0, e_747_0]; try rfl)
  have hSH4 : reducePoly (R i0) N V4 • (1 : ExactMatrix (R i0) N 1 1) - w_582_0 =
      reduceMatrix (R i0) N 1 1 (fun _ _ ↦ M4) := by
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.one_apply_eq, smul_eq_mul, mul_one, reduceMatrix]
    exact hV4
  rw [hSH4, hK4, hDI, hYI] at hcol4
  obtain ⟨ν4, hν4, hc4⟩ := column_noiseQ (Qq := (QqZ (i0 : Int))) (y := β • rowOf w_339_0 0 + rowOf w_339_0 1 - actionValueC (actionConst' (cpZ (i0 : Int)) (kdiv i0)) x w_350_0 + e')
    4 hM4 heb hKb4 heg he1 hew (X0I := topRows DI) (X1I := bottomRows DI)
    (fun r c i ↦ hDIb _ c i) (fun r c i ↦ hDIb _ c i) hex hYIb hesm4 (out := w_799_0) (by rw [hcol4]; try rfl)
  have hcol5 := column_valueR (Qq := (QqZ (i0 : Int))) (c' := (cpZ (i0 : Int))) (k := actionConst' (cpZ (i0 : Int)) (kdiv i0)) L hL hLx (by inst_num) (by rw [hLb]; norm_num) (by rw [hLd]; norm_num) (by rw [hLd])
    (p := (pp i0)) (Mx := (Mx i0)) (R_eq_pMx i0 hi0) (by inst_num) LX hLX hLXx (action_cert i0 hi0)
    (G := GQ1_6) (H := w_582_0) h335_0 (z := w_340_0) (A' := w_339_0) (by rw [e_340_0, e_336_0]) h341_0 h342_0 hGQ1_6
    h351_0 (by rw [e_353_0, e_352_0]) h354_0 h355_0 hGM01_4 hRQ1_4 (comb := w_530_0)
    (by rw [e_530_0, e_357_0, e_343_0, e_356_0]) (by decide : ((5 : Int)).toNat < 6) h802_0 hg hc₁ hcx hfw hb
    (Kpre := w_800_0) hpre5 hsm5 (out := w_853_0) (by rw [e_853_0, e_852_0, e_851_0, e_801_0]; try rfl)
  have hSH5 : reducePoly (R i0) N V5 • (1 : ExactMatrix (R i0) N 1 1) - w_582_0 =
      reduceMatrix (R i0) N 1 1 (fun _ _ ↦ M5) := by
    funext r c
    obtain rfl : r = 0 := Subsingleton.elim _ _
    obtain rfl : c = 0 := Subsingleton.elim _ _
    simp only [Matrix.sub_apply, Matrix.smul_apply, Matrix.one_apply_eq, smul_eq_mul, mul_one, reduceMatrix]
    exact hV5
  rw [hSH5, hK5, hDI, hYI] at hcol5
  obtain ⟨ν5, hν5, hc5⟩ := column_noiseQ (Qq := (QqZ (i0 : Int))) (y := β • rowOf w_339_0 0 + rowOf w_339_0 1 - actionValueC (actionConst' (cpZ (i0 : Int)) (kdiv i0)) x w_350_0 + e')
    5 hM5 heb hKb5 heg he1 hew (X0I := topRows DI) (X1I := bottomRows DI)
    (fun r c i ↦ hDIb _ c i) (fun r c i ↦ hDIb _ c i) hex hYIb hesm5 (out := w_853_0) (by rw [hcol5]; try rfl)
  have hz := concat_six h637_0 h691_0 h745_0 h799_0 h854_0
  intro k i
  refine switch_residue (Qq := (Qq i0)) (by inst_num) (by inst_num) (by inst_num) h855_0
    (y := β • rowOf w_339_0 0 + rowOf w_339_0 1 - actionValueC (actionConst' (cpZ (i0 : Int)) (kdiv i0)) x w_350_0 + e')
    (ν := ![ν0 0 0, ν1 0 0, ν2 0 0, ν3 0 0, ν4 0 0, ν5 0 0]) (fun k ↦ ?_) (fun k j ↦ ?_) k i
  · simp only [Qq_cast]
    rw [hz k]
    fin_cases k
    · exact hc0
    · exact hc1
    · exact hc2
    · exact hc3
    · exact hc4
    · exact hc5
  · fin_cases k
    · have := hν0 0 0 j
      show 2 * |(ν0 0 0).coeff j| < (((Qq i0) : Nat) : Int)
      rw [Int.abs_eq_natAbs]; omega
    · have := hν1 0 0 j
      show 2 * |(ν1 0 0).coeff j| < (((Qq i0) : Nat) : Int)
      rw [Int.abs_eq_natAbs]; omega
    · have := hν2 0 0 j
      show 2 * |(ν2 0 0).coeff j| < (((Qq i0) : Nat) : Int)
      rw [Int.abs_eq_natAbs]; omega
    · have := hν3 0 0 j
      show 2 * |(ν3 0 0).coeff j| < (((Qq i0) : Nat) : Int)
      rw [Int.abs_eq_natAbs]; omega
    · have := hν4 0 0 j
      show 2 * |(ν4 0 0).coeff j| < (((Qq i0) : Nat) : Int)
      rw [Int.abs_eq_natAbs]; omega
    · have := hν5 0 0 j
      show 2 * |(ν5 0 0).coeff j| < (((Qq i0) : Nat) : Int)
      rw [Int.abs_eq_natAbs]; omega

end RefreshTopProof.Part_I
