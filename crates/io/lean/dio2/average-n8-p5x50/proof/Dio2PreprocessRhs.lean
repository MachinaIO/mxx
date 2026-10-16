import Dio2Online0Fresh
import Dio2Online0State
import Gen_W_Stage_preprocess_0_0_Scopes
import Gen_W_Stage_preprocess_1_1_Scopes
import W_Stage_preprocess_0_0_Facts
import W_Stage_preprocess_1_1_Facts

/-!
# The preprocessed selector right-hand sides

Every selector in the two preprocess stages is encrypted with an explicit zero error, not a
Gaussian sample. Its online cell therefore has target
`T j = reducePoly q 8 (message coefficient exponent * unitPair sp j) • G` and `BB = 0`.
The coefficient and exponent families are genuine external inputs; no assumption about their
intended selector values is hidden here. `selector_rhs_meaning` substitutes a supplied meaning
`message coefficient exponent = ν`, if an online sampler requires a particular selector `ν`.

The quantified stage lemmas cover every `rhs/RhsId(k)/family.{0,1}` output. In `W_Claim`, step 0
exports these as preprocess wires `79_{6,7}`, `83_{6,7}`, `36_{2,3}`, `116_{6,7}`,
`120_{6,7}`, `161_{6,7}`, `165_{6,7}` for IDs 0 through 6. The corresponding online wires are
`1233/1234`, `1237/1238`, `298/308`, `27/28`, `31/32`, `1179/1180`, `1183/1184`.
Step 1 exports wires `75_{6,7}`, `79_{6,7}`, `33_{2,3}` for IDs 0 through 2, consumed online as
`146/147`, `150/151`, `72/82` respectively.
IDs 0/1 supply the smudge PRFs, ID 2 supplies the state LUT input, IDs 3/4 supply the sigma
sampler, and IDs 5/6 supply fresh noise. The same rebased gadget can be used for every cell:
`gadgetMatrixRuns_unique` and `centeredRebaseRuns_unique` identify the computed gadgets.
-/

set_option maxHeartbeats 16000000
set_option maxRecDepth 16384

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen

namespace PreprocessRhs

/-- An error-free selector column encrypts the external plaintext `coefficient X^exponent`
times its unit, with the cancellation performed at the working modulus. -/
theorem selector_rhs_exact {q n columns : Nat} {sp unit : ErrorPoly n}
    {secret unitMatrix : ExactMatrix q n 1 1}
    (hs : secret = fun _ _ ↦ reducePoly q n sp)
    (hu : unitMatrix = fun _ _ ↦ reducePoly q n unit)
    (coefficient exponent : Int) (gadget top : ExactMatrix q n 1 columns) :
    reducePoly q n sp • top +
        matrixAdd (matrixSub (matrixMulScalarRight
          (matrixMulScalarRight (matrixMulScalarLeft unitMatrix gadget) (matrixPolynomial [1]))
          (multiplyMonomial (liftInteger coefficient) exponent))
          (matrixMulScalarLeft secret top)) (0 : ExactMatrix q n 1 columns) =
      reducePoly q n (message coefficient exponent * unit) • gadget +
        reduceMatrix q n 1 columns (0 : ErrorMatrix n 1 columns) := by
  rw [hs, hu, multiplyMonomial_liftInteger]
  simp only [matrixMulScalarRight_eq, matrixMulScalarLeft_eq, matrixPolynomial_reduceG]
  funext row column
  simp only [matrixAdd, matrixSub, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply,
    Matrix.zero_apply, smul_eq_mul, map_mul, Int.cast_one, map_one, reduceMatrix,
    map_zero]
  ring

/-- Both error-free selector columns, with precisely the unit pair used by `cell_flat`. -/
theorem selector_rhs_exact_pair {q n columns : Nat} {sp : ErrorPoly n}
    {secret one : ExactMatrix q n 1 1}
    (hs : secret = fun _ _ ↦ reducePoly q n sp)
    (hone : one = fun _ _ ↦ reducePoly q n 1)
    (coefficient exponent : Int) (gadget top0 top1 : ExactMatrix q n 1 columns) :
    ∀ j : Fin 2, reducePoly q n sp • ![top0, top1] j +
        ![matrixAdd (matrixSub (matrixMulScalarRight
            (matrixMulScalarRight (matrixMulScalarLeft secret gadget) (matrixPolynomial [1]))
            (multiplyMonomial (liftInteger coefficient) exponent))
            (matrixMulScalarLeft secret top0)) (0 : ExactMatrix q n 1 columns),
          matrixAdd (matrixSub (matrixMulScalarRight
            (matrixMulScalarRight (matrixMulScalarLeft one gadget) (matrixPolynomial [1]))
            (multiplyMonomial (liftInteger coefficient) exponent))
            (matrixMulScalarLeft secret top1)) (0 : ExactMatrix q n 1 columns)] j =
      reducePoly q n (message coefficient exponent * unitPair sp j) • gadget +
        reduceMatrix q n 1 columns (0 : ErrorMatrix n 1 columns) := by
  intro j
  fin_cases j
  · exact selector_rhs_exact hs hs coefficient exponent gadget top0
  · exact selector_rhs_exact hs hone coefficient exponent gadget top1

/-- The explicit zero error discharges every online selector's `heB` hypothesis with `BB = 0`. -/
theorem selector_zero_bound {n columns count : Nat} :
    ∀ (_ : Fin count) (_ : Fin 2), CoeffBound (0 : ErrorMatrix n 1 columns) 0 := by
  intro i j
  simp [CoeffBound]

/-- Only the interpretation of the externally supplied coefficient/exponent families is assumed:
substituting it yields `hrhs` and `heB` together, uniformly over the cell family. -/
theorem selector_rhs_meaning {q n columns count : Nat} {sp : ErrorPoly n}
    {coefficient exponent : Fin count → Int} {ν : Fin count → ErrorPoly n}
    {top bot : Fin count → Fin 2 → ExactMatrix q n 1 columns}
    {gadget : ExactMatrix q n 1 columns}
    (hrhs : ∀ i j, reducePoly q n sp • top i j + bot i j =
      reducePoly q n (message (coefficient i) (exponent i) * unitPair sp j) • gadget +
        reduceMatrix q n 1 columns (0 : ErrorMatrix n 1 columns))
    (hmeaning : ∀ i, message (coefficient i) (exponent i) = ν i) :
    ∃ error : Fin count → Fin 2 → ErrorMatrix n 1 columns,
      (∀ i j, CoeffBound (error i j) 0) ∧ ∀ i j,
        reducePoly q n sp • top i j + bot i j =
          reducePoly q n (ν i * unitPair sp j) • gadget +
            reduceMatrix q n 1 columns (error i j) := by
  refine ⟨fun _ _ ↦ 0, selector_zero_bound, ?_⟩
  intro i j
  simpa only [hmeaning i] using hrhs i j

end PreprocessRhs

namespace Preprocess0

open W_Stage_preprocess_0_0

/-- Step 0, `RhsId(0)` (smudge first bucket): every selector RHS has the explicit external
plaintext target and zero error. This discharges `hrhs`; `selector_zero_bound` discharges `heB`. -/
theorem rhs0 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) {sp : ErrorPoly 8}
    (hs : generatedRoot.w_1_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 sp)
    {gadgetSource gadget : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 gadgetSource)
    (hGr : centeredRebaseRuns gadgetSource gadget) :
    ∀ (i : Fin 72000) (j : Fin 2), reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_0_0 ctx) modulus_0 25 rhsPre1
          (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_79_6 ctx i,
          generatedRoot.w_79_7 ctx i] j =
      reducePoly modulus_0 8
          (message (generatedRoot.w_77_0 ctx i) (generatedRoot.w_78_0 ctx i) * unitPair sp j) •
        gadget +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  intro i
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, -, -, -, -, -, -, -, -, -,
    -, -, -, -, -, -, -, -, -, -, htop0, -, -, -, -, -, -, -, hgadget, hrebase, -, -, -, htop1,
    -, -, -, -, -, -, -, hout⟩ :=
    parallel_generatedRoot_79.sem (F.c3.h79_7 i)
  obtain rfl := gadgetMatrixRuns_unique hgadget hG
  obtain rfl := centeredRebaseRuns_unique hrebase hGr
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, -, -, -, -, hbot0, hbot1, -⟩ := hout
  have htop := prf_tops htop0 htop1
  rw [← htop]
  simpa only [hbot0, hbot1] using
    PreprocessRhs.selector_rhs_exact_pair hs reduce_one_matrixG
      (generatedRoot.w_77_0 ctx i) (generatedRoot.w_78_0 ctx i) v20 v11 v24

end Preprocess0

namespace Preprocess0

open W_Stage_preprocess_0_0

/-- Step 0, `RhsId(1)` (smudge second bucket): every selector RHS has the explicit external
plaintext target and zero error. This discharges `hrhs`; `selector_zero_bound` discharges `heB`. -/
theorem rhs1 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) {sp : ErrorPoly 8}
    (hs : generatedRoot.w_1_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 sp)
    {gadgetSource gadget : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 gadgetSource)
    (hGr : centeredRebaseRuns gadgetSource gadget) :
    ∀ (i : Fin 72000) (j : Fin 2), reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_0_0 ctx) modulus_0 25 rhsPre1b
          (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_83_6 ctx i,
          generatedRoot.w_83_7 ctx i] j =
      reducePoly modulus_0 8
          (message (generatedRoot.w_81_0 ctx i) (generatedRoot.w_82_0 ctx i) * unitPair sp j) •
        gadget +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  intro i
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, -, -,
    -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, htop0, -, -, -, -, -,
    -, -, -, -, -, -, -, -, hgadget, hrebase, -, -, -, htop1, -, -, -, -, -, -, -, hout⟩ :=
    parallel_generatedRoot_83.sem (F.c3.h83_7 i)
  obtain rfl := gadgetMatrixRuns_unique hgadget hG
  obtain rfl := centeredRebaseRuns_unique hrebase hGr
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, -, -, -, -, hbot0, hbot1, -⟩ := hout
  have htop := prf_tops htop0 htop1
  rw [← htop]
  simpa only [hbot0, hbot1] using
    PreprocessRhs.selector_rhs_exact_pair hs reduce_one_matrixG
      (generatedRoot.w_81_0 ctx i) (generatedRoot.w_82_0 ctx i) v24 v13 v28

end Preprocess0

namespace Preprocess0

open W_Stage_preprocess_0_0

/-- Step 0, `RhsId(2)` (state LUT input): every selector RHS has the explicit external
plaintext target and zero error. This discharges `hrhs`; `selector_zero_bound` discharges `heB`. -/
theorem rhs2 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) {sp : ErrorPoly 8}
    (hs : generatedRoot.w_1_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 sp)
    {gadgetSource gadget : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 gadgetSource)
    (hGr : centeredRebaseRuns gadgetSource gadget) :
    ∀ (i : Fin 1) (j : Fin 2), reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_0_0 ctx) modulus_0 25 rhsPreY
          (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_36_2 ctx i,
          generatedRoot.w_36_3 ctx i] j =
      reducePoly modulus_0 8
          (message (generatedRoot.w_34_0 ctx i) (generatedRoot.w_35_0 ctx i) * unitPair sp j) •
        gadget +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  intro i
  obtain ⟨v0, v1, v2, v3, htop0, htop1, hgadget, hrebase, hout⟩ :=
    parallel_generatedRoot_36.sem (F.c1.h36_3 i)
  obtain rfl := gadgetMatrixRuns_unique hgadget hG
  obtain rfl := centeredRebaseRuns_unique hrebase hGr
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, hbot0, hbot1, -⟩ := hout
  have htop := prf_tops htop0 htop1
  rw [← htop]
  simpa only [hbot0, hbot1] using
    PreprocessRhs.selector_rhs_exact_pair hs reduce_one_matrixG
      (generatedRoot.w_34_0 ctx i) (generatedRoot.w_35_0 ctx i) v3 v0 v1

end Preprocess0

namespace Preprocess0

open W_Stage_preprocess_0_0

/-- Step 0, `RhsId(3)` (sigma first bucket): every selector RHS has the explicit external
plaintext target and zero error. This discharges `hrhs`; `selector_zero_bound` discharges `heB`. -/
theorem rhs3 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) {sp : ErrorPoly 8}
    (hs : generatedRoot.w_1_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 sp)
    {gadgetSource gadget : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 gadgetSource)
    (hGr : centeredRebaseRuns gadgetSource gadget) :
    ∀ (i : Fin 16) (j : Fin 2), reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_0_0 ctx) modulus_0 25 rhsPre0
          (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_116_6 ctx i,
          generatedRoot.w_116_7 ctx i] j =
      reducePoly modulus_0 8
          (message (generatedRoot.w_114_0 ctx i) (generatedRoot.w_115_0 ctx i) * unitPair sp j) •
        gadget +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  intro i
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, -, -, -, -, -, -, -, -, -,
    -, -, -, -, -, -, -, -, -, -, htop0, -, -, -, -, -, -, -, hgadget, hrebase, -, -, -, htop1,
    -, -, -, -, -, -, -, hout⟩ :=
    parallel_generatedRoot_116.sem (F.c4.h116_7 i)
  obtain rfl := gadgetMatrixRuns_unique hgadget hG
  obtain rfl := centeredRebaseRuns_unique hrebase hGr
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, -, -, -, -, hbot0, hbot1, -⟩ := hout
  have htop := prf_tops htop0 htop1
  rw [← htop]
  simpa only [hbot0, hbot1] using
    PreprocessRhs.selector_rhs_exact_pair hs reduce_one_matrixG
      (generatedRoot.w_114_0 ctx i) (generatedRoot.w_115_0 ctx i) v20 v11 v24

end Preprocess0

namespace Preprocess0

open W_Stage_preprocess_0_0

/-- Step 0, `RhsId(4)` (sigma second bucket): every selector RHS has the explicit external
plaintext target and zero error. This discharges `hrhs`; `selector_zero_bound` discharges `heB`. -/
theorem rhs4 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) {sp : ErrorPoly 8}
    (hs : generatedRoot.w_1_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 sp)
    {gadgetSource gadget : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 gadgetSource)
    (hGr : centeredRebaseRuns gadgetSource gadget) :
    ∀ (i : Fin 16) (j : Fin 2), reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_0_0 ctx) modulus_0 25 rhsPre0b
          (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_120_6 ctx i,
          generatedRoot.w_120_7 ctx i] j =
      reducePoly modulus_0 8
          (message (generatedRoot.w_118_0 ctx i) (generatedRoot.w_119_0 ctx i) * unitPair sp j) •
        gadget +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  intro i
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, -, -,
    -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, htop0, -, -, -, -, -,
    -, -, -, -, -, -, -, -, hgadget, hrebase, -, -, -, htop1, -, -, -, -, -, -, -, hout⟩ :=
    parallel_generatedRoot_120.sem (F.c5.h120_7 i)
  obtain rfl := gadgetMatrixRuns_unique hgadget hG
  obtain rfl := centeredRebaseRuns_unique hrebase hGr
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, -, -, -, -, hbot0, hbot1, -⟩ := hout
  have htop := prf_tops htop0 htop1
  rw [← htop]
  simpa only [hbot0, hbot1] using
    PreprocessRhs.selector_rhs_exact_pair hs reduce_one_matrixG
      (generatedRoot.w_118_0 ctx i) (generatedRoot.w_119_0 ctx i) v24 v13 v28

end Preprocess0

namespace Preprocess0

open W_Stage_preprocess_0_0

/-- Step 0, `RhsId(5)` (fresh-noise first bucket): every selector RHS has the explicit external
plaintext target and zero error. This discharges `hrhs`; `selector_zero_bound` discharges `heB`. -/
theorem rhs5 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) {sp : ErrorPoly 8}
    (hs : generatedRoot.w_1_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 sp)
    {gadgetSource gadget : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 gadgetSource)
    (hGr : centeredRebaseRuns gadgetSource gadget) :
    ∀ (i : Fin 14400) (j : Fin 2), reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_0_0 ctx) modulus_0 25 rhsPreF0
          (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_161_6 ctx i,
          generatedRoot.w_161_7 ctx i] j =
      reducePoly modulus_0 8
          (message (generatedRoot.w_159_0 ctx i) (generatedRoot.w_160_0 ctx i) * unitPair sp j) •
        gadget +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  intro i
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, -, -, -, -, -, -, -, -, -,
    -, -, -, -, -, -, -, -, -, -, htop0, -, -, -, -, -, -, -, hgadget, hrebase, -, -, -, htop1,
    -, -, -, -, -, -, -, hout⟩ :=
    parallel_generatedRoot_161.sem (F.c6.h161_7 i)
  obtain rfl := gadgetMatrixRuns_unique hgadget hG
  obtain rfl := centeredRebaseRuns_unique hrebase hGr
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, -, -, -, -, hbot0, hbot1, -⟩ := hout
  have htop := prf_tops htop0 htop1
  rw [← htop]
  simpa only [hbot0, hbot1] using
    PreprocessRhs.selector_rhs_exact_pair hs reduce_one_matrixG
      (generatedRoot.w_159_0 ctx i) (generatedRoot.w_160_0 ctx i) v20 v11 v24

end Preprocess0

namespace Preprocess0

open W_Stage_preprocess_0_0

/-- Step 0, `RhsId(6)` (fresh-noise second bucket): every selector RHS has the explicit external
plaintext target and zero error. This discharges `hrhs`; `selector_zero_bound` discharges `heB`. -/
theorem rhs6 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) {sp : ErrorPoly 8}
    (hs : generatedRoot.w_1_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 sp)
    {gadgetSource gadget : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 gadgetSource)
    (hGr : centeredRebaseRuns gadgetSource gadget) :
    ∀ (i : Fin 14400) (j : Fin 2), reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_0_0 ctx) modulus_0 25 rhsPreF0b
          (prfTopTags prfA0 prfB0 i) j +
        ![generatedRoot.w_165_6 ctx i,
          generatedRoot.w_165_7 ctx i] j =
      reducePoly modulus_0 8
          (message (generatedRoot.w_163_0 ctx i) (generatedRoot.w_164_0 ctx i) * unitPair sp j) •
        gadget +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  intro i
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, -, -,
    -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, htop0, -, -, -, -, -,
    -, -, -, -, -, -, -, -, hgadget, hrebase, -, -, -, htop1, -, -, -, -, -, -, -, hout⟩ :=
    parallel_generatedRoot_165.sem (F.c6.h165_7 i)
  obtain rfl := gadgetMatrixRuns_unique hgadget hG
  obtain rfl := centeredRebaseRuns_unique hrebase hGr
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, -, -, -, -, hbot0, hbot1, -⟩ := hout
  have htop := prf_tops htop0 htop1
  rw [← htop]
  simpa only [hbot0, hbot1] using
    PreprocessRhs.selector_rhs_exact_pair hs reduce_one_matrixG
      (generatedRoot.w_163_0 ctx i) (generatedRoot.w_164_0 ctx i) v24 v13 v28

end Preprocess0

namespace Preprocess1

open W_Stage_preprocess_1_1

/-- Step 1, `RhsId(0)` (smudge first bucket): every selector RHS has the explicit external
plaintext target and zero error. This discharges `hrhs`; `selector_zero_bound` discharges `heB`. -/
theorem rhs0 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) {sp : ErrorPoly 8}
    (hs : generatedRoot.w_1_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 sp)
    {gadgetSource gadget : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 gadgetSource)
    (hGr : centeredRebaseRuns gadgetSource gadget) :
    ∀ (i : Fin 16) (j : Fin 2), reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_0_0 ctx) modulus_0 25 rhsPre1
          (prfTopTags prfA1 prfB1 i) j +
        ![generatedRoot.w_75_6 ctx i,
          generatedRoot.w_75_7 ctx i] j =
      reducePoly modulus_0 8
          (message (generatedRoot.w_73_0 ctx i) (generatedRoot.w_74_0 ctx i) * unitPair sp j) •
        gadget +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  intro i
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, -, -, -, -, -, -, -, -, -,
    -, -, -, -, -, -, -, -, -, -, htop0, -, -, -, -, -, -, -, hgadget, hrebase, -, -, -, htop1,
    -, -, -, -, -, -, -, hout⟩ :=
    parallel_generatedRoot_75.sem (F.c3.h75_7 i)
  obtain rfl := gadgetMatrixRuns_unique hgadget hG
  obtain rfl := centeredRebaseRuns_unique hrebase hGr
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, -, -, -, -, hbot0, hbot1, -⟩ := hout
  have htop := prf_tops htop0 htop1
  rw [← htop]
  simpa only [hbot0, hbot1] using
    PreprocessRhs.selector_rhs_exact_pair hs reduce_one_matrixG
      (generatedRoot.w_73_0 ctx i) (generatedRoot.w_74_0 ctx i) v20 v11 v24

end Preprocess1

namespace Preprocess1

open W_Stage_preprocess_1_1

/-- Step 1, `RhsId(1)` (smudge second bucket): every selector RHS has the explicit external
plaintext target and zero error. This discharges `hrhs`; `selector_zero_bound` discharges `heB`. -/
theorem rhs1 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) {sp : ErrorPoly 8}
    (hs : generatedRoot.w_1_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 sp)
    {gadgetSource gadget : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 gadgetSource)
    (hGr : centeredRebaseRuns gadgetSource gadget) :
    ∀ (i : Fin 16) (j : Fin 2), reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_0_0 ctx) modulus_0 25 rhsPre1b
          (prfTopTags prfA1 prfB1 i) j +
        ![generatedRoot.w_79_6 ctx i,
          generatedRoot.w_79_7 ctx i] j =
      reducePoly modulus_0 8
          (message (generatedRoot.w_77_0 ctx i) (generatedRoot.w_78_0 ctx i) * unitPair sp j) •
        gadget +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  intro i
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, -, -,
    -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, -, htop0, -, -, -, -, -,
    -, -, -, -, -, -, -, -, hgadget, hrebase, -, -, -, htop1, -, -, -, -, -, -, -, hout⟩ :=
    parallel_generatedRoot_79.sem (F.c3.h79_7 i)
  obtain rfl := gadgetMatrixRuns_unique hgadget hG
  obtain rfl := centeredRebaseRuns_unique hrebase hGr
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, -, -, -, -, hbot0, hbot1, -⟩ := hout
  have htop := prf_tops htop0 htop1
  rw [← htop]
  simpa only [hbot0, hbot1] using
    PreprocessRhs.selector_rhs_exact_pair hs reduce_one_matrixG
      (generatedRoot.w_77_0 ctx i) (generatedRoot.w_78_0 ctx i) v24 v13 v28

end Preprocess1

namespace Preprocess1

open W_Stage_preprocess_1_1

/-- Step 1, `RhsId(2)` (state LUT input): every selector RHS has the explicit external
plaintext target and zero error. This discharges `hrhs`; `selector_zero_bound` discharges `heB`. -/
theorem rhs2 {backend : BackendContext} {hashModel : HashModel} {params : Params}
    (ctx : generatedRoot.Ctx backend hashModel params) {outputs : _}
    (F : generatedRoot.Facts ctx outputs) {sp : ErrorPoly 8}
    (hs : generatedRoot.w_1_0 ctx = fun _ _ ↦ reducePoly modulus_0 8 sp)
    {gadgetSource gadget : ExactMatrix modulus_0 8 1 25}
    (hG : gadgetMatrixRuns backend 4096 25 gadgetSource)
    (hGr : centeredRebaseRuns gadgetSource gadget) :
    ∀ (i : Fin 1) (j : Fin 2), reducePoly modulus_0 8 sp •
        topPairG hashModel (generatedRoot.w_0_0 ctx) modulus_0 25 rhsPreY
          (prfTopTags prfA1 prfB1 i) j +
        ![generatedRoot.w_33_2 ctx i,
          generatedRoot.w_33_3 ctx i] j =
      reducePoly modulus_0 8
          (message (generatedRoot.w_31_0 ctx i) (generatedRoot.w_32_0 ctx i) * unitPair sp j) •
        gadget +
        reduceMatrix modulus_0 8 1 25 (0 : ErrorMatrix 8 1 25) := by
  intro i
  obtain ⟨v0, v1, v2, v3, htop0, htop1, hgadget, hrebase, hout⟩ :=
    parallel_generatedRoot_33.sem (F.c0.h33_3 i)
  obtain rfl := gadgetMatrixRuns_unique hgadget hG
  obtain rfl := centeredRebaseRuns_unique hrebase hGr
  simp only [Prod.mk.injEq] at hout
  obtain ⟨-, -, hbot0, hbot1, -⟩ := hout
  have htop := prf_tops htop0 htop1
  rw [← htop]
  simpa only [hbot0, hbot1] using
    PreprocessRhs.selector_rhs_exact_pair hs reduce_one_matrixG
      (generatedRoot.w_31_0 ctx i) (generatedRoot.w_32_0 ctx i) v3 v0 v1

end Preprocess1

end Dio2Proof
