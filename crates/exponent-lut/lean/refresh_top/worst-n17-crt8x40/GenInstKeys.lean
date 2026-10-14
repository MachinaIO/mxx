import GenBridge_Prf
import GenBridge_LutKey
import ExponentRefresh

/-!
# Key parts of the PRF cell bundles (WP D)

`PrfCell0.keys` / `PrfCell1.keys` (and `_1` for a Program in its own stages): the keys of an
online PRF cell (the second conjunct of `PrfCell0.rel` / `PrfCell1.rel`, GenBridge_Prf) need no
semantic hypothesis; they are the same
`cellKeyG` forms the preprocessing key cells (`PrfKeyCell0.rel` / `PrfKeyCell1.rel`) conclude, so
the online and preprocessing keys agree when their arguments do.
-/

namespace RefreshTopProof.GenIface

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen

variable {n : Nat}

set_option maxHeartbeats 8000000 in
theorem PrfCell0.keys (hn : 0 < n) {q q' p p' m m' m'' K bound : Nat} (hq : q = p * q')
    (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p)
    {backend : BackendContext} {hashModel : HashModel} {base digits base' digits' : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0) (hm : m = 1 * layout.digitCount)
    (layout' : RegularLayout q') (hlayout' : backend.regularLayout q' n = some layout')
    (hexact' : layout'.droppedModuli = 0) (hm' : m' = 1 * layout'.digitCount)
    {lo hi a b : Int} {rhsPre : Blob} {basis : List Nat} {i : Nat}
    {idx : Int} {x0 xA00 xA10 x1 xA01 xA11 : ExactMatrix q n 1 m}
    {As00 As10 As01 As11 cs0 cs1 : ExactMatrix q n 1 m'} {pay : Fin K → ExactMatrix q n 1 1}
    {seed : ByteArray} {bot0 bot1 : ExactMatrix q' n 1 m''}
    {outputs : ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' ×
      ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' × Unit}
    (h : PrfCell0.Comp q n m' m' m' m' m'' m'' m m q K q q' q' q' m'' q' m'' q' m'' q' m'' q' m''
      q' m'' base digits bound m' lo hi m' m' base digits bound m' m' m' a b rhsPre base' digits'
      bound m'' basis a b rhsPre base' digits' bound m'' basis m'' basis m'' basis m'' basis m''
      basis backend hashModel i
      (idx, x0, xA00, xA10, x1, xA01, xA11, As00, As10, cs0, pay, As01, As11, cs1, seed, bot0, bot1,
        ()) outputs)
    (k : Fin K) (hidx : idx = ((k : Nat) : Int)) :
    Enc.A outputs = cellKeyG backend base digits base' digits' basis
        ![![xA00, xA01], ![xA10, xA11]] ![![As00, As01], ![As10, As11]]
        (topPairG hashModel seed q' m'' rhsPre (prfTopTags a b i)) ![bot0, bot1] := by
  obtain ⟨v0, h⟩ := h
  obtain ⟨v1, h⟩ := h
  obtain ⟨v2, h⟩ := h
  obtain ⟨v3, h⟩ := h
  obtain ⟨v4, h⟩ := h
  obtain ⟨v5, h⟩ := h
  obtain ⟨v6, h⟩ := h
  obtain ⟨v7, h⟩ := h
  obtain ⟨v8, h⟩ := h
  obtain ⟨v9, h⟩ := h
  obtain ⟨v10, h⟩ := h
  obtain ⟨v11, h⟩ := h
  obtain ⟨v12, h⟩ := h
  obtain ⟨v13, h⟩ := h
  obtain ⟨v14, h⟩ := h
  obtain ⟨v15, h⟩ := h
  obtain ⟨v16, h⟩ := h
  obtain ⟨v17, h⟩ := h
  obtain ⟨v18, h⟩ := h
  obtain ⟨v19, h⟩ := h
  obtain ⟨v20, h⟩ := h
  obtain ⟨v21, h⟩ := h
  obtain ⟨v22, h⟩ := h
  obtain ⟨v23, h⟩ := h
  obtain ⟨v24, h⟩ := h
  obtain ⟨v25, h⟩ := h
  obtain ⟨v26, h⟩ := h
  obtain ⟨v27, h⟩ := h
  obtain ⟨v28, h⟩ := h
  obtain ⟨v29, h⟩ := h
  obtain ⟨v30, h⟩ := h
  obtain ⟨v31, h⟩ := h
  obtain ⟨v32, h⟩ := h
  obtain ⟨v33, h⟩ := h
  obtain ⟨v34, h⟩ := h
  obtain ⟨v35, h⟩ := h
  obtain ⟨v36, h⟩ := h
  obtain ⟨v37, h⟩ := h
  obtain ⟨v38, h⟩ := h
  obtain ⟨v39, h⟩ := h
  obtain ⟨v40, h⟩ := h
  obtain ⟨v41, h⟩ := h
  obtain ⟨v42, h⟩ := h
  obtain ⟨c1, h⟩ := h
  obtain ⟨c2, h⟩ := h
  obtain ⟨c3, h⟩ := h
  obtain ⟨c4, h⟩ := h
  obtain ⟨c5, h⟩ := h
  obtain ⟨c6, h⟩ := h
  obtain ⟨c7, h⟩ := h
  obtain ⟨c8, h⟩ := h
  obtain ⟨c9, h⟩ := h
  obtain ⟨c10, h⟩ := h
  obtain ⟨c11, h⟩ := h
  obtain ⟨c12, h⟩ := h
  obtain ⟨c13, h⟩ := h
  obtain ⟨c14, h⟩ := h
  obtain ⟨c15, h⟩ := h
  obtain ⟨c16, h⟩ := h
  obtain ⟨c17, h⟩ := h
  obtain ⟨c18, h⟩ := h
  obtain ⟨c19, h⟩ := h
  obtain ⟨c20, h⟩ := h
  obtain ⟨c21, h⟩ := h
  obtain ⟨c22, h⟩ := h
  obtain ⟨c23, h⟩ := h
  obtain ⟨c24, h⟩ := h
  obtain ⟨c25, h⟩ := h
  obtain ⟨c26, h⟩ := h
  obtain ⟨c27, h⟩ := h
  obtain ⟨c28, h⟩ := h
  obtain ⟨c29, h⟩ := h
  obtain ⟨c30, h⟩ := h
  obtain ⟨c31, h⟩ := h
  obtain ⟨c32, h⟩ := h
  obtain ⟨c33, h⟩ := h
  obtain ⟨c34, h⟩ := h
  obtain ⟨c35, h⟩ := h
  obtain ⟨c36, h⟩ := h
  obtain ⟨c37, h⟩ := h
  obtain ⟨c38, h⟩ := h
  obtain ⟨c39, h⟩ := h
  obtain ⟨c40, h⟩ := h
  obtain ⟨c41, h⟩ := h
  obtain ⟨c42, h⟩ := h
  obtain ⟨c43, h⟩ := h
  obtain ⟨c44, h⟩ := h
  obtain ⟨c45, h⟩ := h
  subst h
  have htop := prf_tops c26 c32
  have hXs : ∀ j, concatRows (![![As00, As01], ![As10, As11]] 0 j) (![![As00, As01], ![As10, As11]] 1 j)
      (![v5, v14] j) := fun j ↦ by fin_cases j; exacts [c6, c17]
  have hD' : ∀ j, gadgetDecomposeRuns backend base digits (![v5, v14] j) (![v6, v15] j) :=
    fun j ↦ by fin_cases j; exacts [c7, c18]
  have hD : ∀ j, compactCenteredRebaseRuns bound (![v6, v15] j) (![v7, v16] j) :=
    fun j ↦ by fin_cases j; exacts [c8, c19]
  have hA0 : ∀ j, sliceMatrix (matrixMul v4 (![v7, v16] j)) 1 2 0 m' (![![v10, v18], ![v11, v19]] 0 j) :=
    fun j ↦ by fin_cases j; exacts [c13, c21]
  have hA1 : ∀ j, sliceMatrix (matrixMul v4 (![v7, v16] j)) 2 3 0 m' (![![v10, v18], ![v11, v19]] 1 j) :=
    fun j ↦ by fin_cases j; exacts [c14, c22]
  have hXb : ∀ j, concatRows (![v23, v29] j) (![bot0, bot1] j) (![v24, v30] j) :=
    fun j ↦ by fin_cases j; exacts [c27, c33]
  have hE' : ∀ j, gadgetDecomposeRuns backend base' digits' (![v24, v30] j) (![v25, v31] j) :=
    fun j ↦ by fin_cases j; exacts [c28, c34]
  have hE : ∀ j, compactCenteredRebaseRuns bound (![v25, v31] j) (![v26, v32] j) :=
    fun j ↦ by fin_cases j; exacts [c29, c35]
  have hM0 : ∀ j, sliceMatrix (matrixMul v22 (![v26, v32] j)) 1 2 0 m'' (![![v27, v33], ![v35, v37]] 0 j) :=
    fun j ↦ by fin_cases j; exacts [c30, c36]
  have hM1 : ∀ j, sliceMatrix (matrixMul v22 (![v26, v32] j)) 2 3 0 m'' (![![v27, v33], ![v35, v37]] 1 j) :=
    fun j ↦ by fin_cases j; exacts [c38, c40]
  have hA'' : ∀ a j, blockModSwitchRuns basis 1 (![![v27, v33], ![v35, v37]] a j)
      (![![v28, v34], ![v36, v38]] a j) := fun a j ↦ by
    fin_cases a <;> fin_cases j; exacts [c31, c37, c39, c41]
  have hM := pubKeyG_of_stacks hn hq1 layout hlayout hexact hm
    (A := ![![xA00, xA01], ![xA10, xA11]]) c1 c2 c3 c4 c5 hXs hD' hD hA0 hA1
  have hK := privKeyG_of_stacks hn hq' layout' hlayout' hexact' hm'
    (c := ![matrixAdd v8 (matrixMulScalarRight cs0 v9), matrixAdd v17 (matrixMulScalarRight cs1 v9)])
    (A := ![![v10, v18], ![v11, v19]]) c15 c16 c23 c24 c25 hXb hE' hE hM0 hM1 hA''
  rw [← htop]
  exact cellKeyG_of_parts hM hK


set_option maxHeartbeats 8000000 in
theorem PrfCell1.keys (hn : 0 < n) {Q q q' pA p p' mA m m' m'' K W bound : Nat}
    (hQ : Q = pA * q) (hQ1 : 1 < Q) (hpA : 0 < pA) (hq : q = p * q') (hq1 : 1 < q)
    (hq' : 1 < q') (hp : 0 < p)
    {backend : BackendContext} {hashModel : HashModel}
    {baseA digitsA base digits base' digits' : Int}
    (layoutA : RegularLayout Q) (hlayoutA : backend.regularLayout Q n = some layoutA)
    (hexactA : layoutA.droppedModuli = 0)
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0) (hm : m = 1 * layout.digitCount)
    (layout' : RegularLayout q') (hlayout' : backend.regularLayout q' n = some layout')
    (hexact' : layout'.droppedModuli = 0) (hm' : m' = 1 * layout'.digitCount)
    {pAi lo hi loK hiK P a b : Int} (hpAi : (pA : Int) = pAi) {rhsPre : Blob}
    {basisA basis : List Nat} {i : Nat}
    {idx : Int} {x0 xA00 xA10 x1 xA01 xA11 : ExactMatrix Q n 1 mA}
    {As00 As10 As01 As11 cs0 cs1 : Fin W → ExactMatrix q n 1 m'}
    {pay : Fin K → ExactMatrix Q n 1 1} {seed : ByteArray} {bot0 bot1 : ExactMatrix q' n 1 m''}
    {outputs : ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' ×
      ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' × Unit}
    (h : PrfCell1.Comp Q n m Q q m' m' m' m' m'' m'' mA mA W q W W K Q W q W W q' q' q' m'' q' m''
      q' m'' q' m'' q' m'' q' m'' q q
      base digits baseA digitsA [pAi] basisA base digits baseA digitsA [pAi] basisA basisA basisA
      basisA basisA lo P hi P P lo P hi P P base digits bound m' lo P hi P P loK hiK m' m'
      lo P hi P P lo P hi P P base digits bound m' lo P hi P P m' m'
      a b rhsPre base' digits' bound m'' basis a b rhsPre base' digits' bound m'' basis
      m'' basis m'' basis m'' basis m'' basis
      backend hashModel i
      (idx, x0, xA00, xA10, x1, xA01, xA11, As00, As10, cs0, pay, As01, As11, cs1, seed, bot0, bot1,
        ()) outputs)
    (k : Fin K) (hidx : idx = ((k : Nat) : Int))
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {GR : ExactMatrix Q n 1 m} (hGR : centeredRebaseRuns G GR)
    :
    ∃ o : Fin W, ((o : Nat) : Int) = Int.fdiv (Int.ofNat i) P ∧
      Enc.A outputs = cellKeyG backend base digits base' digits' basis
        (alignKeyG backend baseA digitsA basisA ![![xA00, xA01], ![xA10, xA11]]
          (reducePoly Q n ((pA : Int) : ErrorPoly n) • GR))
        ![![As00 o, As01 o], ![As10 o, As11 o]]
        (topPairG hashModel seed q' m'' rhsPre (prfTopTags a b i)) ![bot0, bot1] := by
  subst hpAi
  obtain ⟨v0, h⟩ := h
  obtain ⟨v1, h⟩ := h
  obtain ⟨v2, h⟩ := h
  obtain ⟨v3, h⟩ := h
  obtain ⟨v4, h⟩ := h
  obtain ⟨v5, h⟩ := h
  obtain ⟨v6, h⟩ := h
  obtain ⟨v7, h⟩ := h
  obtain ⟨v8, h⟩ := h
  obtain ⟨v9, h⟩ := h
  obtain ⟨v10, h⟩ := h
  obtain ⟨v11, h⟩ := h
  obtain ⟨v12, h⟩ := h
  obtain ⟨v13, h⟩ := h
  obtain ⟨v14, h⟩ := h
  obtain ⟨v15, h⟩ := h
  obtain ⟨v16, h⟩ := h
  obtain ⟨v17, h⟩ := h
  obtain ⟨v18, h⟩ := h
  obtain ⟨v19, h⟩ := h
  obtain ⟨v20, h⟩ := h
  obtain ⟨v21, h⟩ := h
  obtain ⟨v22, h⟩ := h
  obtain ⟨v23, h⟩ := h
  obtain ⟨v24, h⟩ := h
  obtain ⟨v25, h⟩ := h
  obtain ⟨v26, h⟩ := h
  obtain ⟨v27, h⟩ := h
  obtain ⟨v28, h⟩ := h
  obtain ⟨v29, h⟩ := h
  obtain ⟨v30, h⟩ := h
  obtain ⟨v31, h⟩ := h
  obtain ⟨v32, h⟩ := h
  obtain ⟨v33, h⟩ := h
  obtain ⟨v34, h⟩ := h
  obtain ⟨v35, h⟩ := h
  obtain ⟨v36, h⟩ := h
  obtain ⟨v37, h⟩ := h
  obtain ⟨v38, h⟩ := h
  obtain ⟨v39, h⟩ := h
  obtain ⟨v40, h⟩ := h
  obtain ⟨v41, h⟩ := h
  obtain ⟨v42, h⟩ := h
  obtain ⟨v43, h⟩ := h
  obtain ⟨v44, h⟩ := h
  obtain ⟨v45, h⟩ := h
  obtain ⟨v46, h⟩ := h
  obtain ⟨v47, h⟩ := h
  obtain ⟨v48, h⟩ := h
  obtain ⟨v49, h⟩ := h
  obtain ⟨v50, h⟩ := h
  obtain ⟨v51, h⟩ := h
  obtain ⟨v52, h⟩ := h
  obtain ⟨v53, h⟩ := h
  obtain ⟨v54, h⟩ := h
  obtain ⟨v55, h⟩ := h
  obtain ⟨v56, h⟩ := h
  obtain ⟨v57, h⟩ := h
  obtain ⟨v58, h⟩ := h
  obtain ⟨v59, h⟩ := h
  obtain ⟨v60, h⟩ := h
  obtain ⟨v61, h⟩ := h
  obtain ⟨c1, h⟩ := h
  obtain ⟨c2, h⟩ := h
  obtain ⟨c3, h⟩ := h
  obtain ⟨c4, h⟩ := h
  obtain ⟨c5, h⟩ := h
  obtain ⟨c6, h⟩ := h
  obtain ⟨c7, h⟩ := h
  obtain ⟨c8, h⟩ := h
  obtain ⟨c9, h⟩ := h
  obtain ⟨c10, h⟩ := h
  obtain ⟨c11, h⟩ := h
  obtain ⟨c12, h⟩ := h
  obtain ⟨c13, h⟩ := h
  obtain ⟨c14, h⟩ := h
  obtain ⟨c15, h⟩ := h
  obtain ⟨c16, h⟩ := h
  obtain ⟨c17, h⟩ := h
  obtain ⟨c18, h⟩ := h
  obtain ⟨c19, h⟩ := h
  obtain ⟨c20, h⟩ := h
  obtain ⟨c21, h⟩ := h
  obtain ⟨c22, h⟩ := h
  obtain ⟨c23, h⟩ := h
  obtain ⟨c24, h⟩ := h
  obtain ⟨c25, h⟩ := h
  obtain ⟨c26, h⟩ := h
  obtain ⟨c27, h⟩ := h
  obtain ⟨c28, h⟩ := h
  obtain ⟨c29, h⟩ := h
  obtain ⟨c30, h⟩ := h
  obtain ⟨c31, h⟩ := h
  obtain ⟨c32, h⟩ := h
  obtain ⟨c33, h⟩ := h
  obtain ⟨c34, h⟩ := h
  obtain ⟨c35, h⟩ := h
  obtain ⟨c36, h⟩ := h
  obtain ⟨c37, h⟩ := h
  obtain ⟨c38, h⟩ := h
  obtain ⟨c39, h⟩ := h
  obtain ⟨c40, h⟩ := h
  obtain ⟨c41, h⟩ := h
  obtain ⟨c42, h⟩ := h
  obtain ⟨c43, h⟩ := h
  obtain ⟨c44, h⟩ := h
  obtain ⟨c45, h⟩ := h
  obtain ⟨c46, h⟩ := h
  obtain ⟨c47, h⟩ := h
  obtain ⟨c48, h⟩ := h
  obtain ⟨c49, h⟩ := h
  obtain ⟨c50, h⟩ := h
  obtain ⟨c51, h⟩ := h
  obtain ⟨c52, h⟩ := h
  obtain ⟨c53, h⟩ := h
  obtain ⟨c54, h⟩ := h
  obtain ⟨c55, h⟩ := h
  obtain ⟨c56, h⟩ := h
  obtain ⟨c57, h⟩ := h
  obtain ⟨c58, h⟩ := h
  obtain ⟨c59, h⟩ := h
  obtain ⟨c60, h⟩ := h
  obtain ⟨c61, h⟩ := h
  obtain ⟨c62, h⟩ := h
  obtain ⟨c63, h⟩ := h
  obtain ⟨c64, h⟩ := h
  obtain ⟨c65, h⟩ := h
  obtain ⟨c66, h⟩ := h
  obtain ⟨c67, h⟩ := h
  obtain ⟨c68, h⟩ := h
  obtain ⟨c69, h⟩ := h
  obtain ⟨c70, h⟩ := h
  obtain ⟨c71, h⟩ := h
  obtain ⟨c72, h⟩ := h
  obtain ⟨c73, h⟩ := h
  obtain ⟨c74, h⟩ := h
  obtain ⟨c75, h⟩ := h
  obtain ⟨c76, h⟩ := h
  subst h
  have e0 := gadgetMatrixRuns_unique c1 hG
  have e4 := gadgetMatrixRuns_unique c5 hG
  subst v0 v4
  have e1 := centeredRebaseRuns_unique c2 hGR
  have e5 := centeredRebaseRuns_unique c6 hGR
  subst v1 v5
  have e6 := gadgetDecomposeRuns_unique c7 c3
  subst v6
  obtain ⟨o, ho, rfl⟩ := c20
  obtain ⟨o1, ho1, rfl⟩ := c23
  obtain ⟨o2, ho2, rfl⟩ := c30
  obtain ⟨o3, ho3, rfl⟩ := c41
  obtain ⟨o4, ho4, rfl⟩ := c44
  obtain ⟨o5, ho5, rfl⟩ := c51
  have e1o : o1 = o := Fin.ext (by have := ho1.trans ho.symm; omega)
  subst o1
  have e2o : o2 = o := Fin.ext (by have := ho2.trans ho.symm; omega)
  subst o2
  have e3o : o3 = o := Fin.ext (by have := ho3.trans ho.symm; omega)
  subst o3
  have e4o : o4 = o := Fin.ext (by have := ho4.trans ho.symm; omega)
  subst o4
  have e5o : o5 = o := Fin.ext (by have := ho5.trans ho.symm; omega)
  subst o5
  obtain ⟨pos, hpos, rfl⟩ := c33
  have epos : pos = k := Fin.ext (by omega)
  subst pos
  have hscaled : matrixMulScalarLeft (matrixPolynomial [((pA : Nat) : Int)]) GR =
      reducePoly Q n ((pA : Int) : ErrorPoly n) • GR := by
    rw [matrixMulScalarLeft_eq, matrixPolynomial_reduceG]
  have htop := prf_tops c57 c63
  have hXs : ∀ j, concatRows (![![As00 o, As01 o], ![As10 o, As11 o]] 0 j)
      (![![As00 o, As01 o], ![As10 o, As11 o]] 1 j) (![v19, v32] j) :=
    fun j ↦ by fin_cases j; exacts [c24, c45]
  have hD' : ∀ j, gadgetDecomposeRuns backend base digits (![v19, v32] j) (![v20, v33] j) :=
    fun j ↦ by fin_cases j; exacts [c25, c46]
  have hD : ∀ j, compactCenteredRebaseRuns bound (![v20, v33] j) (![v21, v34] j) :=
    fun j ↦ by fin_cases j; exacts [c26, c47]
  have hA0 : ∀ j, sliceMatrix (matrixMul v16 (![v21, v34] j)) 1 2 0 m'
      (![![v26, v37], ![v27, v38]] 0 j) := fun j ↦ by fin_cases j; exacts [c35, c52]
  have hA1 : ∀ j, sliceMatrix (matrixMul v16 (![v21, v34] j)) 2 3 0 m'
      (![![v26, v37], ![v27, v38]] 1 j) := fun j ↦ by fin_cases j; exacts [c36, c53]
  have hXb : ∀ j, concatRows (![v42, v48] j) (![bot0, bot1] j) (![v43, v49] j) :=
    fun j ↦ by fin_cases j; exacts [c58, c64]
  have hE' : ∀ j, gadgetDecomposeRuns backend base' digits' (![v43, v49] j) (![v44, v50] j) :=
    fun j ↦ by fin_cases j; exacts [c59, c65]
  have hE : ∀ j, compactCenteredRebaseRuns bound (![v44, v50] j) (![v45, v51] j) :=
    fun j ↦ by fin_cases j; exacts [c60, c66]
  have hM0 : ∀ j, sliceMatrix (matrixMul v41 (![v45, v51] j)) 1 2 0 m''
      (![![v46, v52], ![v54, v56]] 0 j) := fun j ↦ by fin_cases j; exacts [c61, c67]
  have hM1 : ∀ j, sliceMatrix (matrixMul v41 (![v45, v51] j)) 2 3 0 m''
      (![![v46, v52], ![v54, v56]] 1 j) := fun j ↦ by fin_cases j; exacts [c69, c71]
  have hA'' : ∀ a j, blockModSwitchRuns basis 1 (![![v46, v52], ![v54, v56]] a j)
      (![![v47, v53], ![v55, v57]] a j) := fun a j ↦ by
    fin_cases a <;> fin_cases j; exacts [c62, c68, c70, c72]
  refine ⟨o, ho, ?_⟩
  have hal := alignKeyG_of_runs (A := ![![xA00, xA01], ![xA10, xA11]])
    (A' := ![![v7, v12], ![v8, v13]]) (basis := basisA) c3
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [c8, c13, c9, c14])
  rw [hscaled] at hal
  have hM := pubKeyG_of_stacks hn hq1 layout hlayout hexact hm
    (A := ![![v7, v12], ![v8, v13]]) c10 c11 c15 c16 c17 hXs hD' hD hA0 hA1
  have hK := privKeyG_of_stacks hn hq' layout' hlayout' hexact' hm'
    (c := ![matrixAdd v22 (matrixMulScalarRight (cs0 o) v25),
      matrixAdd v35 (matrixMulScalarRight (cs1 o) v25)])
    (A := ![![v26, v37], ![v27, v38]]) c37 c38 c54 c55 c56 hXb hE' hE hM0 hM1 hA''
  rw [← htop, ← hal]
  exact cellKeyG_of_parts hM hK


/-- The label keys of member `k` as their four components (`labelKeyG`). -/
theorem labelKeysG_eq {q m : Nat} (model : HashModel) (key : ByteArray) (pre : Blob) (k : Int) :
    labelKeysG (n := n) model key q m pre k =
      ![![labelKeyG model q n m key pre k 0, labelKeyG model q n m key pre k 1],
        ![labelKeyG model q n m key pre k 2, labelKeyG model q n m key pre k 3]] := by
  funext a b
  fin_cases a <;> fin_cases b <;> rfl

set_option maxHeartbeats 8000000 in
/-- `PrfCell0.keys` for a Program in its own stages (variant 1, `PrfCell0.rel_1`): the input
encoding's keys are the member's hash-sampled label keys `labelKeysG`. -/
theorem PrfCell0.keys_1 (hn : 0 < n) {q q' p p' m m' m'' bound E : Nat} (hq : q = p * q')
    (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p)
    {backend : BackendContext} {hashModel : HashModel} {base digits base' digits' : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0) (hm : m = 1 * layout.digitCount)
    (layout' : RegularLayout q') (hlayout' : backend.regularLayout q' n = some layout')
    (hexact' : layout'.droppedModuli = 0) (hm' : m' = 1 * layout'.digitCount)
    {lo P a b : Int} {pre rhsPre : Blob} {basis : List Nat} {i : Nat}
    {idx : Int} {x0 x1 : ExactMatrix q n 1 m} {key : ByteArray}
    {As00 As10 As01 As11 cs0 cs1 : ExactMatrix q n 1 m'} {pay : ExactMatrix q n 1 1}
    {bot0 bot1 : ExactMatrix q' n 1 m''}
    {outputs : ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' ×
      ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' × Unit}
    (h : PrfCell0.Comp_1 q n m' m' m' m' m'' m'' m m q q q' q' q' m'' q' m'' q' m'' q' m'' q' m''
      q' m'' lo P E pre lo P E pre lo P E pre lo P E pre base digits bound m' m' m' base digits
      bound m' m' m' a b rhsPre base' digits' bound m'' basis a b rhsPre base' digits' bound m''
      basis m'' basis m'' basis m'' basis m'' basis backend hashModel i
      (idx, x0, key, x1, As00, As10, cs0, pay, As01, As11, cs1, bot0, bot1, ()) outputs) :
    Enc.A outputs = cellKeyG backend base digits base' digits' basis
        (labelKeysG hashModel key q m pre idx) ![![As00, As01], ![As10, As11]]
        (topPairG hashModel key q' m'' rhsPre (prfTopTags a b i)) ![bot0, bot1] := by
  obtain ⟨v0, h⟩ := h
  obtain ⟨v1, h⟩ := h
  obtain ⟨v2, h⟩ := h
  obtain ⟨v3, h⟩ := h
  obtain ⟨v4, h⟩ := h
  obtain ⟨v5, h⟩ := h
  obtain ⟨v6, h⟩ := h
  obtain ⟨v7, h⟩ := h
  obtain ⟨v8, h⟩ := h
  obtain ⟨v9, h⟩ := h
  obtain ⟨v10, h⟩ := h
  obtain ⟨v11, h⟩ := h
  obtain ⟨v12, h⟩ := h
  obtain ⟨v13, h⟩ := h
  obtain ⟨v14, h⟩ := h
  obtain ⟨v15, h⟩ := h
  obtain ⟨v16, h⟩ := h
  obtain ⟨v17, h⟩ := h
  obtain ⟨v18, h⟩ := h
  obtain ⟨v19, h⟩ := h
  obtain ⟨v20, h⟩ := h
  obtain ⟨v21, h⟩ := h
  obtain ⟨v22, h⟩ := h
  obtain ⟨v23, h⟩ := h
  obtain ⟨v24, h⟩ := h
  obtain ⟨v25, h⟩ := h
  obtain ⟨v26, h⟩ := h
  obtain ⟨v27, h⟩ := h
  obtain ⟨v28, h⟩ := h
  obtain ⟨v29, h⟩ := h
  obtain ⟨v30, h⟩ := h
  obtain ⟨v31, h⟩ := h
  obtain ⟨v32, h⟩ := h
  obtain ⟨v33, h⟩ := h
  obtain ⟨v34, h⟩ := h
  obtain ⟨v35, h⟩ := h
  obtain ⟨v36, h⟩ := h
  obtain ⟨v37, h⟩ := h
  obtain ⟨v38, h⟩ := h
  obtain ⟨v39, h⟩ := h
  obtain ⟨v40, h⟩ := h
  obtain ⟨v41, h⟩ := h
  obtain ⟨v42, h⟩ := h
  obtain ⟨v43, h⟩ := h
  obtain ⟨v44, h⟩ := h
  obtain ⟨v45, h⟩ := h
  obtain ⟨c1, h⟩ := h
  obtain ⟨c2, h⟩ := h
  obtain ⟨c3, h⟩ := h
  obtain ⟨c4, h⟩ := h
  obtain ⟨c5, h⟩ := h
  obtain ⟨c6, h⟩ := h
  obtain ⟨c7, h⟩ := h
  obtain ⟨c8, h⟩ := h
  obtain ⟨c9, h⟩ := h
  obtain ⟨c10, h⟩ := h
  obtain ⟨c11, h⟩ := h
  obtain ⟨c12, h⟩ := h
  obtain ⟨c13, h⟩ := h
  obtain ⟨c14, h⟩ := h
  obtain ⟨c15, h⟩ := h
  obtain ⟨c16, h⟩ := h
  obtain ⟨c17, h⟩ := h
  obtain ⟨c18, h⟩ := h
  obtain ⟨c19, h⟩ := h
  obtain ⟨c20, h⟩ := h
  obtain ⟨c21, h⟩ := h
  obtain ⟨c22, h⟩ := h
  obtain ⟨c23, h⟩ := h
  obtain ⟨c24, h⟩ := h
  obtain ⟨c25, h⟩ := h
  obtain ⟨c26, h⟩ := h
  obtain ⟨c27, h⟩ := h
  obtain ⟨c28, h⟩ := h
  obtain ⟨c29, h⟩ := h
  obtain ⟨c30, h⟩ := h
  obtain ⟨c31, h⟩ := h
  obtain ⟨c32, h⟩ := h
  obtain ⟨c33, h⟩ := h
  obtain ⟨c34, h⟩ := h
  obtain ⟨c35, h⟩ := h
  obtain ⟨c36, h⟩ := h
  obtain ⟨c37, h⟩ := h
  obtain ⟨c38, h⟩ := h
  obtain ⟨c39, h⟩ := h
  obtain ⟨c40, h⟩ := h
  obtain ⟨c41, h⟩ := h
  obtain ⟨c42, h⟩ := h
  obtain ⟨c43, h⟩ := h
  obtain ⟨c44, h⟩ := h
  obtain ⟨c45, h⟩ := h
  obtain ⟨c46, h⟩ := h
  obtain ⟨c47, h⟩ := h
  obtain ⟨c48, h⟩ := h
  obtain ⟨c49, h⟩ := h
  obtain ⟨c50, h⟩ := h
  obtain ⟨c51, h⟩ := h
  obtain ⟨c52, h⟩ := h
  obtain ⟨c53, h⟩ := h
  obtain ⟨c54, h⟩ := h
  subst h
  have e0 := labelKeyG_of c3
  have e1 := labelKeyG_of c6
  have e4 := labelKeyG_of c11
  have e5 := labelKeyG_of c14
  subst e0 e1 e4 e5
  have htop := prf_tops c35 c41
  have hXs : ∀ j, concatRows (![![As00, As01], ![As10, As11]] 0 j) (![![As00, As01], ![As10, As11]] 1 j)
      (![v9, v17] j) := fun j ↦ by fin_cases j; exacts [c18, c26]
  have hD' : ∀ j, gadgetDecomposeRuns backend base digits (![v9, v17] j) (![v10, v18] j) :=
    fun j ↦ by fin_cases j; exacts [c19, c27]
  have hD : ∀ j, compactCenteredRebaseRuns bound (![v10, v18] j) (![v11, v19] j) :=
    fun j ↦ by fin_cases j; exacts [c20, c28]
  have hA0 : ∀ j, sliceMatrix (matrixMul v8 (![v11, v19] j)) 1 2 0 m' (![![v13, v21], ![v14, v22]] 0 j) :=
    fun j ↦ by fin_cases j; exacts [c22, c30]
  have hA1 : ∀ j, sliceMatrix (matrixMul v8 (![v11, v19] j)) 2 3 0 m' (![![v13, v21], ![v14, v22]] 1 j) :=
    fun j ↦ by fin_cases j; exacts [c23, c31]
  have hXb : ∀ j, concatRows (![v26, v32] j) (![bot0, bot1] j) (![v27, v33] j) :=
    fun j ↦ by fin_cases j; exacts [c36, c42]
  have hE' : ∀ j, gadgetDecomposeRuns backend base' digits' (![v27, v33] j) (![v28, v34] j) :=
    fun j ↦ by fin_cases j; exacts [c37, c43]
  have hE : ∀ j, compactCenteredRebaseRuns bound (![v28, v34] j) (![v29, v35] j) :=
    fun j ↦ by fin_cases j; exacts [c38, c44]
  have hM0 : ∀ j, sliceMatrix (matrixMul v25 (![v29, v35] j)) 1 2 0 m'' (![![v30, v36], ![v38, v40]] 0 j) :=
    fun j ↦ by fin_cases j; exacts [c39, c45]
  have hM1 : ∀ j, sliceMatrix (matrixMul v25 (![v29, v35] j)) 2 3 0 m'' (![![v30, v36], ![v38, v40]] 1 j) :=
    fun j ↦ by fin_cases j; exacts [c47, c49]
  have hA'' : ∀ a j, blockModSwitchRuns basis 1 (![![v30, v36], ![v38, v40]] a j)
      (![![v31, v37], ![v39, v41]] a j) := fun a j ↦ by
    fin_cases a <;> fin_cases j; exacts [c40, c46, c48, c50]
  have hM := pubKeyG_of_stacks hn hq1 layout hlayout hexact hm
    (A := ![![labelKeyG hashModel q n m key pre idx 0, labelKeyG hashModel q n m key pre idx 1], ![labelKeyG hashModel q n m key pre idx 2, labelKeyG hashModel q n m key pre idx 3]]) c7 c8 c15 c16 c17 hXs hD' hD hA0 hA1
  have hK := privKeyG_of_stacks hn hq' layout' hlayout' hexact' hm'
    (c := ![matrixAdd v12 (matrixMulScalarRight cs0 pay), matrixAdd v20 (matrixMulScalarRight cs1 pay)])
    (A := ![![v13, v21], ![v14, v22]]) c24 c25 c32 c33 c34 hXb hE' hE hM0 hM1 hA''
  rw [← htop, labelKeysG_eq]
  exact cellKeyG_of_parts hM hK

set_option maxHeartbeats 8000000 in
/-- `PrfCell1.keys` for a Program in its own stages (variant 1, `PrfCell1.rel_1`). -/
theorem PrfCell1.keys_1 (hn : 0 < n) {Q q q' pA p p' mA m m' m'' W bound E : Nat}
    (hQ : Q = pA * q) (hQ1 : 1 < Q) (hpA : 0 < pA) (hq : q = p * q') (hq1 : 1 < q)
    (hq' : 1 < q') (hp : 0 < p)
    {backend : BackendContext} {hashModel : HashModel}
    {baseA digitsA base digits base' digits' : Int}
    (layoutA : RegularLayout Q) (hlayoutA : backend.regularLayout Q n = some layoutA)
    (hexactA : layoutA.droppedModuli = 0)
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0) (hm : m = 1 * layout.digitCount)
    (layout' : RegularLayout q') (hlayout' : backend.regularLayout q' n = some layout')
    (hexact' : layout'.droppedModuli = 0) (hm' : m' = 1 * layout'.digitCount)
    {pAi lo hi P a b lo64 P64 : Int} (hpAi : (pA : Int) = pAi) {pre rhsPre : Blob}
    {basisA basis : List Nat} {i : Nat}
    {idx : Int} {x0 x1 : ExactMatrix Q n 1 mA} {key : ByteArray}
    {As00 As10 As01 As11 cs0 cs1 : Fin W → ExactMatrix q n 1 m'}
    {pay : ExactMatrix Q n 1 1} {bot0 bot1 : ExactMatrix q' n 1 m''}
    {outputs : ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' ×
      ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' × ExactMatrix q' n 1 m'' × Unit}
    (h : PrfCell1.Comp_1 Q n m Q q m' m' m' m' m'' m'' mA W q W W Q W q W W q' q' q' m'' q' m''
      q' m'' q' m'' q' m'' q' m'' q mA q
      base digits baseA digitsA [pAi] basisA lo64 P64 E pre base digits baseA digitsA [pAi] basisA
      lo64 P64 E pre basisA basisA lo64 P64 E pre basisA lo64 P64 E pre basisA
      lo P hi P P lo P hi P P base digits bound m' lo P hi P P m' m'
      lo P hi P P lo P hi P P base digits bound m' lo P hi P P m' m'
      a b rhsPre base' digits' bound m'' basis a b rhsPre base' digits' bound m'' basis
      m'' basis m'' basis m'' basis m'' basis
      backend hashModel i
      (idx, x0, key, x1, As00, As10, cs0, pay, As01, As11, cs1, bot0, bot1, ()) outputs)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {GR : ExactMatrix Q n 1 m} (hGR : centeredRebaseRuns G GR) :
    ∃ o : Fin W, ((o : Nat) : Int) = Int.fdiv (Int.ofNat i) P ∧
      Enc.A outputs = cellKeyG backend base digits base' digits' basis
        (alignKeyG backend baseA digitsA basisA (labelKeysG hashModel key Q mA pre idx)
          (reducePoly Q n ((pA : Int) : ErrorPoly n) • GR))
        ![![As00 o, As01 o], ![As10 o, As11 o]]
        (topPairG hashModel key q' m'' rhsPre (prfTopTags a b i)) ![bot0, bot1] := by
  subst hpAi
  obtain ⟨v0, h⟩ := h
  obtain ⟨v1, h⟩ := h
  obtain ⟨v2, h⟩ := h
  obtain ⟨v3, h⟩ := h
  obtain ⟨v4, h⟩ := h
  obtain ⟨v5, h⟩ := h
  obtain ⟨v6, h⟩ := h
  obtain ⟨v7, h⟩ := h
  obtain ⟨v8, h⟩ := h
  obtain ⟨v9, h⟩ := h
  obtain ⟨v10, h⟩ := h
  obtain ⟨v11, h⟩ := h
  obtain ⟨v12, h⟩ := h
  obtain ⟨v13, h⟩ := h
  obtain ⟨v14, h⟩ := h
  obtain ⟨v15, h⟩ := h
  obtain ⟨v16, h⟩ := h
  obtain ⟨v17, h⟩ := h
  obtain ⟨v18, h⟩ := h
  obtain ⟨v19, h⟩ := h
  obtain ⟨v20, h⟩ := h
  obtain ⟨v21, h⟩ := h
  obtain ⟨v22, h⟩ := h
  obtain ⟨v23, h⟩ := h
  obtain ⟨v24, h⟩ := h
  obtain ⟨v25, h⟩ := h
  obtain ⟨v26, h⟩ := h
  obtain ⟨v27, h⟩ := h
  obtain ⟨v28, h⟩ := h
  obtain ⟨v29, h⟩ := h
  obtain ⟨v30, h⟩ := h
  obtain ⟨v31, h⟩ := h
  obtain ⟨v32, h⟩ := h
  obtain ⟨v33, h⟩ := h
  obtain ⟨v34, h⟩ := h
  obtain ⟨v35, h⟩ := h
  obtain ⟨v36, h⟩ := h
  obtain ⟨v37, h⟩ := h
  obtain ⟨v38, h⟩ := h
  obtain ⟨v39, h⟩ := h
  obtain ⟨v40, h⟩ := h
  obtain ⟨v41, h⟩ := h
  obtain ⟨v42, h⟩ := h
  obtain ⟨v43, h⟩ := h
  obtain ⟨v44, h⟩ := h
  obtain ⟨v45, h⟩ := h
  obtain ⟨v46, h⟩ := h
  obtain ⟨v47, h⟩ := h
  obtain ⟨v48, h⟩ := h
  obtain ⟨v49, h⟩ := h
  obtain ⟨v50, h⟩ := h
  obtain ⟨v51, h⟩ := h
  obtain ⟨v52, h⟩ := h
  obtain ⟨v53, h⟩ := h
  obtain ⟨v54, h⟩ := h
  obtain ⟨v55, h⟩ := h
  obtain ⟨v56, h⟩ := h
  obtain ⟨v57, h⟩ := h
  obtain ⟨v58, h⟩ := h
  obtain ⟨v59, h⟩ := h
  obtain ⟨v60, h⟩ := h
  obtain ⟨v61, h⟩ := h
  obtain ⟨v62, h⟩ := h
  obtain ⟨v63, h⟩ := h
  obtain ⟨v64, h⟩ := h
  obtain ⟨c1, h⟩ := h
  obtain ⟨c2, h⟩ := h
  obtain ⟨c3, h⟩ := h
  obtain ⟨c4, h⟩ := h
  obtain ⟨c5, h⟩ := h
  obtain ⟨c6, h⟩ := h
  obtain ⟨c7, h⟩ := h
  obtain ⟨c8, h⟩ := h
  obtain ⟨c9, h⟩ := h
  obtain ⟨c10, h⟩ := h
  obtain ⟨c11, h⟩ := h
  obtain ⟨c12, h⟩ := h
  obtain ⟨c13, h⟩ := h
  obtain ⟨c14, h⟩ := h
  obtain ⟨c15, h⟩ := h
  obtain ⟨c16, h⟩ := h
  obtain ⟨c17, h⟩ := h
  obtain ⟨c18, h⟩ := h
  obtain ⟨c19, h⟩ := h
  obtain ⟨c20, h⟩ := h
  obtain ⟨c21, h⟩ := h
  obtain ⟨c22, h⟩ := h
  obtain ⟨c23, h⟩ := h
  obtain ⟨c24, h⟩ := h
  obtain ⟨c25, h⟩ := h
  obtain ⟨c26, h⟩ := h
  obtain ⟨c27, h⟩ := h
  obtain ⟨c28, h⟩ := h
  obtain ⟨c29, h⟩ := h
  obtain ⟨c30, h⟩ := h
  obtain ⟨c31, h⟩ := h
  obtain ⟨c32, h⟩ := h
  obtain ⟨c33, h⟩ := h
  obtain ⟨c34, h⟩ := h
  obtain ⟨c35, h⟩ := h
  obtain ⟨c36, h⟩ := h
  obtain ⟨c37, h⟩ := h
  obtain ⟨c38, h⟩ := h
  obtain ⟨c39, h⟩ := h
  obtain ⟨c40, h⟩ := h
  obtain ⟨c41, h⟩ := h
  obtain ⟨c42, h⟩ := h
  obtain ⟨c43, h⟩ := h
  obtain ⟨c44, h⟩ := h
  obtain ⟨c45, h⟩ := h
  obtain ⟨c46, h⟩ := h
  obtain ⟨c47, h⟩ := h
  obtain ⟨c48, h⟩ := h
  obtain ⟨c49, h⟩ := h
  obtain ⟨c50, h⟩ := h
  obtain ⟨c51, h⟩ := h
  obtain ⟨c52, h⟩ := h
  obtain ⟨c53, h⟩ := h
  obtain ⟨c54, h⟩ := h
  obtain ⟨c55, h⟩ := h
  obtain ⟨c56, h⟩ := h
  obtain ⟨c57, h⟩ := h
  obtain ⟨c58, h⟩ := h
  obtain ⟨c59, h⟩ := h
  obtain ⟨c60, h⟩ := h
  obtain ⟨c61, h⟩ := h
  obtain ⟨c62, h⟩ := h
  obtain ⟨c63, h⟩ := h
  obtain ⟨c64, h⟩ := h
  obtain ⟨c65, h⟩ := h
  obtain ⟨c66, h⟩ := h
  obtain ⟨c67, h⟩ := h
  obtain ⟨c68, h⟩ := h
  obtain ⟨c69, h⟩ := h
  obtain ⟨c70, h⟩ := h
  obtain ⟨c71, h⟩ := h
  obtain ⟨c72, h⟩ := h
  obtain ⟨c73, h⟩ := h
  obtain ⟨c74, h⟩ := h
  obtain ⟨c75, h⟩ := h
  obtain ⟨c76, h⟩ := h
  obtain ⟨c77, h⟩ := h
  obtain ⟨c78, h⟩ := h
  obtain ⟨c79, h⟩ := h
  obtain ⟨c80, h⟩ := h
  obtain ⟨c81, h⟩ := h
  obtain ⟨c82, h⟩ := h
  obtain ⟨c83, h⟩ := h
  obtain ⟨c84, h⟩ := h
  obtain ⟨c85, h⟩ := h
  subst h
  have ex0 := labelKeyG_of c7
  have ex2 := labelKeyG_of c14
  have ex1 := labelKeyG_of c21
  have ex3 := labelKeyG_of c25
  subst ex0 ex1 ex2 ex3
  have e0 := gadgetMatrixRuns_unique c1 hG
  have e4 := gadgetMatrixRuns_unique c8 hG
  subst v0 v5
  have e1 := centeredRebaseRuns_unique c2 hGR
  have e5 := centeredRebaseRuns_unique c9 hGR
  subst v1 v6
  have e6 := gadgetDecomposeRuns_unique c10 c3
  subst v7
  obtain ⟨o, ho, rfl⟩ := c32
  obtain ⟨o1, ho1, rfl⟩ := c35
  obtain ⟨o2, ho2, rfl⟩ := c42
  obtain ⟨o3, ho3, rfl⟩ := c50
  obtain ⟨o4, ho4, rfl⟩ := c53
  obtain ⟨o5, ho5, rfl⟩ := c60
  have e1o : o1 = o := Fin.ext (by have := ho1.trans ho.symm; omega)
  subst o1
  have e2o : o2 = o := Fin.ext (by have := ho2.trans ho.symm; omega)
  subst o2
  have e3o : o3 = o := Fin.ext (by have := ho3.trans ho.symm; omega)
  subst o3
  have e4o : o4 = o := Fin.ext (by have := ho4.trans ho.symm; omega)
  subst o4
  have e5o : o5 = o := Fin.ext (by have := ho5.trans ho.symm; omega)
  subst o5
  have hscaled : matrixMulScalarLeft (matrixPolynomial [((pA : Nat) : Int)]) GR =
      reducePoly Q n ((pA : Int) : ErrorPoly n) • GR := by
    rw [matrixMulScalarLeft_eq, matrixPolynomial_reduceG]
  have htop := prf_tops c66 c72
  have hXs : ∀ j, concatRows (![![As00 o, As01 o], ![As10 o, As11 o]] 0 j)
      (![![As00 o, As01 o], ![As10 o, As11 o]] 1 j) (![v23, v35] j) :=
    fun j ↦ by fin_cases j; exacts [c36, c54]
  have hD' : ∀ j, gadgetDecomposeRuns backend base digits (![v23, v35] j) (![v24, v36] j) :=
    fun j ↦ by fin_cases j; exacts [c37, c55]
  have hD : ∀ j, compactCenteredRebaseRuns bound (![v24, v36] j) (![v25, v37] j) :=
    fun j ↦ by fin_cases j; exacts [c38, c56]
  have hA0 : ∀ j, sliceMatrix (matrixMul v20 (![v25, v37] j)) 1 2 0 m'
      (![![v29, v40], ![v30, v41]] 0 j) := fun j ↦ by fin_cases j; exacts [c44, c61]
  have hA1 : ∀ j, sliceMatrix (matrixMul v20 (![v25, v37] j)) 2 3 0 m'
      (![![v29, v40], ![v30, v41]] 1 j) := fun j ↦ by fin_cases j; exacts [c45, c62]
  have hXb : ∀ j, concatRows (![v45, v51] j) (![bot0, bot1] j) (![v46, v52] j) :=
    fun j ↦ by fin_cases j; exacts [c67, c73]
  have hE' : ∀ j, gadgetDecomposeRuns backend base' digits' (![v46, v52] j) (![v47, v53] j) :=
    fun j ↦ by fin_cases j; exacts [c68, c74]
  have hE : ∀ j, compactCenteredRebaseRuns bound (![v47, v53] j) (![v48, v54] j) :=
    fun j ↦ by fin_cases j; exacts [c69, c75]
  have hM0 : ∀ j, sliceMatrix (matrixMul v44 (![v48, v54] j)) 1 2 0 m''
      (![![v49, v55], ![v57, v59]] 0 j) := fun j ↦ by fin_cases j; exacts [c70, c76]
  have hM1 : ∀ j, sliceMatrix (matrixMul v44 (![v48, v54] j)) 2 3 0 m''
      (![![v49, v55], ![v57, v59]] 1 j) := fun j ↦ by fin_cases j; exacts [c78, c80]
  have hA'' : ∀ a j, blockModSwitchRuns basis 1 (![![v49, v55], ![v57, v59]] a j)
      (![![v50, v56], ![v58, v60]] a j) := fun a j ↦ by
    fin_cases a <;> fin_cases j; exacts [c71, c77, c79, c81]
  refine ⟨o, ho, ?_⟩
  have hal := alignKeyG_of_runs (A := ![![labelKeyG hashModel Q n mA key pre idx 0, labelKeyG hashModel Q n mA key pre idx 1], ![labelKeyG hashModel Q n mA key pre idx 2, labelKeyG hashModel Q n mA key pre idx 3]])
    (A' := ![![v8, v15], ![v10, v17]]) (basis := basisA) c3
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [c11, c22, c15, c26])
  rw [hscaled] at hal
  have hM := pubKeyG_of_stacks hn hq1 layout hlayout hexact hm
    (A := ![![v8, v15], ![v10, v17]]) c16 c17 c27 c28 c29 hXs hD' hD hA0 hA1
  have hK := privKeyG_of_stacks hn hq' layout' hlayout' hexact' hm'
    (c := ![matrixAdd v26 (matrixMulScalarRight (cs0 o) v28),
      matrixAdd v38 (matrixMulScalarRight (cs1 o) v28)])
    (A := ![![v29, v40], ![v30, v41]]) c46 c47 c63 c64 c65 hXb hE' hE hM0 hM1 hA''
  rw [← htop, labelKeysG_eq, ← hal]
  exact cellKeyG_of_parts hM hK

/-! ## Keys of the online LUT loops and smudge members -/

open BrLut BrLutKey

/-- The keys of a range sum of key blocks are the range sum of their tuples. -/
theorem keyTupleG_sum {q m : Nat} (K : Nat → Fin 2 → Fin 2 → ExactMatrix q n 1 m) (c : Nat) :
    keyTupleG (∑ i ∈ Finset.range c, K i) = ∑ i ∈ Finset.range c, keyTupleG (K i) := by
  induction c with
  | zero => rfl
  | succ c ih => rw [Finset.sum_range_succ, Finset.sum_range_succ, keyTupleG_add, ih]

set_option maxHeartbeats 64000000 in
/-- **The carried switching ciphertexts of one LUT loop iteration**: the body reads member `i + 1`
of the four switching families into the next state (no semantic hypothesis). -/
theorem LutRoundStep.switch {n q q' m m' W : Nat} {backend : BackendContext} {hashModel : HashModel}
    {base digits : Int} {tab : List Int} {key : ByteArray} {level : Int} {tag : Blob}
    {hf0 hf1 : Fin W → ExactMatrix q n 1 m} {x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m}
    {f00 f01 f10 f11 : Fin W → ExactMatrix q' n 1 m'} {i : Nat} {cur nxt : LoopStateG n q q' m'}
    {bound : Nat} {basis : List Nat} {mw : Int} {p0 p1 p8 p9 p50 p51 p52 p53 p54 p55 p56 p57 : Int}
    (h : LutRoundStep.Comp q n m' m' m' m' m' m' m' m' m' m' q' q' W m W m m m W q' n m' W q' n m'
      W q' n m' W q' n m' q' q' p0 p1 level tag tab level tag tab p8 p9 level tag tab level tag tab
      tab tab tab tab tab tab base digits bound mw basis mw basis base digits bound mw mw base digits
      bound mw basis mw basis base digits bound mw mw mw mw mw mw p50 p51 p52 p53 p54 p55 p56 p57
      backend hashModel i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, cur.2.2.2.2.2.2.1,
        cur.2.2.2.2.2.2.2.1, cur.2.2.2.2.2.2.2.2.1, cur.2.2.2.2.2.2.2.2.2.1, hf0, key, hf1,
        x0, x1, x2, x3, x4, x5, f00, f01, f10, f11, ()) nxt) :
    ∃ hi : i + 1 < W, nxt.switch = fun k j ↦ fam4 f00 f01 f10 f11 k j ⟨i + 1, hi⟩ := by
  refine Exists.elim h (fun v0 h ↦ Exists.elim h (fun v1 h ↦ Exists.elim h (fun v2 h ↦ Exists.elim h (fun v3 h ↦ Exists.elim h (fun v4 h ↦ Exists.elim h (fun v5 h ↦ Exists.elim h (fun v6 h ↦ Exists.elim h (fun v7 h ↦ Exists.elim h (fun v8 h ↦ Exists.elim h (fun v9 h ↦ Exists.elim h (fun v10 h ↦ Exists.elim h (fun v11 h ↦ Exists.elim h (fun v12 h ↦ Exists.elim h (fun v13 h ↦ Exists.elim h (fun v14 h ↦ Exists.elim h (fun v15 h ↦ Exists.elim h (fun v16 h ↦ Exists.elim h (fun v17 h ↦ Exists.elim h (fun v18 h ↦ Exists.elim h (fun v19 h ↦ Exists.elim h (fun v20 h ↦ Exists.elim h (fun v21 h ↦ Exists.elim h (fun v22 h ↦ Exists.elim h (fun v23 h ↦ Exists.elim h (fun v24 h ↦ Exists.elim h (fun v25 h ↦ Exists.elim h (fun v26 h ↦ Exists.elim h (fun v27 h ↦ Exists.elim h (fun v28 h ↦ Exists.elim h (fun v29 h ↦ Exists.elim h (fun v30 h ↦ Exists.elim h (fun v31 h ↦ Exists.elim h (fun v32 h ↦ Exists.elim h (fun v33 h ↦ Exists.elim h (fun v34 h ↦ Exists.elim h (fun v35 h ↦ Exists.elim h (fun v36 h ↦ Exists.elim h (fun v37 h ↦ Exists.elim h (fun v38 h ↦ Exists.elim h (fun v39 h ↦ Exists.elim h (fun v40 h ↦ Exists.elim h (fun v41 h ↦ Exists.elim h (fun v42 h ↦ Exists.elim h (fun v43 h ↦ Exists.elim h (fun v44 h ↦ Exists.elim h (fun v45 h ↦ Exists.elim h (fun v46 h ↦ Exists.elim h (fun v47 h ↦ Exists.elim h (fun v48 h ↦ Exists.elim h (fun v49 h ↦ Exists.elim h (fun v50 h ↦ Exists.elim h (fun v51 h ↦ Exists.elim h (fun v52 h ↦ Exists.elim h (fun v53 h ↦ ?_))))))))))))))))))))))))))))))))))))))))))))))))))))))
  obtain ⟨c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, rfl⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_succ i c56
  obtain ⟨_, rfl⟩ := familyGet_succ i c59
  obtain ⟨_, rfl⟩ := familyGet_succ i c62
  obtain ⟨_, rfl⟩ := familyGet_succ i c65
  refine ⟨hi, ?_⟩
  funext k j
  fin_cases k <;> fin_cases j <;> rfl

/-- **The key step of a LUT branch loop body** `child`: iteration `i` adds the branch keys
`branchKeyG … σ_i A (lutAG … σ_i) cur.switch` (`σ_i = selectInt i tab`, `A = inA x0 … x5`) and
carries member `i + 1` of the switching families. -/
def LutKeyStepHyp {q q' m m' W : Nat} (child : Nat → StepIn n q q' m m' W → LoopStateG n q q' m' → Prop)
    (hf0 : Fin W → ExactMatrix q n 1 m) (key : ByteArray) (hf1 : Fin W → ExactMatrix q n 1 m)
    (x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m) (f00 f01 f10 f11 : Fin W → ExactMatrix q' n 1 m')
    (backend : BackendContext) (base digits : Int) (basis : List Nat) (hashModel : HashModel)
    (tag : Blob) (level : Int) (tab : List Int) : Prop :=
  ∀ i (cur nxt : LoopStateG n q q' m'),
    child i (stepIn cur hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11) nxt →
    nxt.key = cur.key + branchKeyG backend base digits basis (MxxIR.selectInt (Int.ofNat i) tab).toNat
      (inA x0 x1 x2 x3 x4 x5) (lutAG hashModel key q m tag level (MxxIR.selectInt (Int.ofNat i) tab))
      cur.switch ∧
    ∃ hi : i + 1 < W, nxt.switch = fun k j ↦ fam4 f00 f01 f10 f11 k j ⟨i + 1, hi⟩

/-- The key step of every iteration of a branch loop whose body bundles to `LutRoundStep.Comp`
(`hcomp`: the child scope's adapter, as for `LutRoundStep.hstep`). -/
theorem LutRoundStep.keystep {n q q' m m' W : Nat} {backend : BackendContext} {hashModel : HashModel}
    {base digits : Int} {tab : List Int} {key : ByteArray} {level : Int} {tag : Blob}
    {hf0 hf1 : Fin W → ExactMatrix q n 1 m} {x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m}
    {f00 f01 f10 f11 : Fin W → ExactMatrix q' n 1 m'}
    {child : Nat → StepIn n q q' m m' W → LoopStateG n q q' m' → Prop}
    {bound : Nat} {basis : List Nat} {mw : Int} {p0 p1 p8 p9 p50 p51 p52 p53 p54 p55 p56 p57 : Int}
    (hcomp : ∀ i (cur nxt : LoopStateG n q q' m'),
      child i (stepIn cur hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11) nxt →
      LutRoundStep.Comp q n m' m' m' m' m' m' m' m' m' m' q' q' W m W m m m W q' n m' W q' n m'
        W q' n m' W q' n m' q' q' p0 p1 level tag tab level tag tab p8 p9 level tag tab level tag
        tab tab tab tab tab tab tab base digits bound mw basis mw basis base digits bound mw mw base
        digits bound mw basis mw basis base digits bound mw mw mw mw mw mw p50 p51 p52 p53 p54 p55
        p56 p57 backend hashModel i
        (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, cur.2.2.2.2.2.2.1,
          cur.2.2.2.2.2.2.2.1, cur.2.2.2.2.2.2.2.2.1, cur.2.2.2.2.2.2.2.2.2.1, hf0, key, hf1,
          x0, x1, x2, x3, x4, x5, f00, f01, f10, f11, ()) nxt)
    (hmw : mw = (m' : Int)) (hn : 0 < n) (hq' : 1 < q') (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    (hm : m = 1 * layout.digitCount) :
    LutKeyStepHyp child hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11 backend base digits basis
      hashModel tag level tab :=
  fun i cur nxt hc ↦ ⟨LutRoundStep.keys (hcomp i cur nxt hc) hmw hn hq' layout hlayout hexact hm,
    LutRoundStep.switch (hcomp i cur nxt hc)⟩

set_option maxHeartbeats 64000000 in
/-- **The keys of a smudge member** (the key part of `SmudgeMember.rel`, no semantic hypothesis):
the four keys of the online member are `SmudgeKeyMember.rel`'s form — the summed branch keys
`lutKeysG` of the `L + 1` branches (`L` looped, the last one unrolled) on the input keys
`inA x0 … x5`, moved down, times the rebased digits of the weight `c P^(i mod d)`, placed at
`X^(⌊i/d⌋ mod N)`. -/
theorem SmudgeMember.keys {n q q' m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits baseG digitsG : Int} {tab : List Int} {key : ByteArray}
    {level : Int} {tag : Blob} {hf0 hf1 : Fin W → ExactMatrix q n 1 m}
    {x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m} {f00 f01 f10 f11 : Fin W → ExactMatrix q' n 1 m'}
    {child : Nat → StepIn n q q' m m' W → LoopStateG n q q' m' → Prop} {i : Nat} {ℓ : Int}
    {bound boundG : Nat} {basis basis' : List Nat} {mw D NN c P : Int}
    {outputs : ExactMatrix q' n 1 1 × ExactMatrix q' n 1 1 × ExactMatrix q' n 1 1 ×
      ExactMatrix q' n 1 1 × ExactMatrix q' n 1 1 × ExactMatrix q' n 1 1 × Unit}
    (h : SmudgeMember.Comp q n m' q' m' q' m' W m W m m m W W W W m' q' m' q' q' q'
      0 0 0 0 ℓ ℓ level tag ℓ tab level tag ℓ tab ℓ level tag ℓ tab level tag ℓ tab
      ℓ tab ℓ tab ℓ tab ℓ tab ℓ tab ℓ tab
      base digits bound mw basis mw basis base digits bound mw basis'
      baseG digitsG [c] P D boundG
      base digits bound mw basis mw basis base digits bound mw basis' mw basis' mw basis' mw mw
      basis' mw mw basis' D NN D NN D NN D NN D NN D NN
      backend hashModel i child (f00, f01, f10, f11, hf0, key, hf1, x0, x1, x2, x3, x4, x5, ())
      outputs)
    (hmw : mw = (m' : Int)) (d : Nat) (hD : D = (d : Int)) (N : Nat) (hNN : NN = (N : Int))
    (L : Nat) (hℓ : ℓ = (L : Int)) (hL : L + 1 = W) (hn : 0 < n) (hq' : 1 < q')
    (layout : RegularLayout q') (hlayout : backend.regularLayout q' n = some layout)
    (hexact : layout.droppedModuli = 0) (hm : m = 1 * layout.digitCount)
    (hchild : LutKeyStepHyp child hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11 backend base digits
      basis hashModel tag level tab) :
    ((outputs.1, outputs.2.1, outputs.2.2.1, outputs.2.2.2.1, ()) : KeysG n q' 1) =
      keyPlaceMulG (rdecG (inner := m') backend baseG digitsG
        (matrixPolynomial [c * P ^ (i % d)] : ExactMatrix q' n 1 1))
        ((i / d % N : Nat) : Int)
        (keyBmsG basis' (∑ k ∈ Finset.range (L + 1), lutKeysG backend base digits basis hashModel key tag
          level (sigN tab) (inA x0 x1 x2 x3 x4 x5) (fam4 f00 f01 f10 f11) k)) := by
  subst hmw hD hNN hℓ hL
  refine Exists.elim h (fun v0 h ↦ Exists.elim h (fun v1 h ↦ Exists.elim h (fun v2 h ↦ Exists.elim h (fun v3 h ↦ Exists.elim h (fun v4 h ↦ Exists.elim h (fun v5 h ↦ Exists.elim h (fun v6 h ↦ Exists.elim h (fun v7 h ↦ Exists.elim h (fun v8 h ↦ Exists.elim h (fun v9 h ↦ Exists.elim h (fun v10 h ↦ Exists.elim h (fun v11 h ↦ Exists.elim h (fun v12 h ↦ Exists.elim h (fun v13 h ↦ Exists.elim h (fun v14 h ↦ Exists.elim h (fun v15 h ↦ Exists.elim h (fun v16 h ↦ Exists.elim h (fun v17 h ↦ Exists.elim h (fun v18 h ↦ Exists.elim h (fun v19 h ↦ Exists.elim h (fun v20 h ↦ Exists.elim h (fun v21 h ↦ Exists.elim h (fun v22 h ↦ Exists.elim h (fun v23 h ↦ Exists.elim h (fun v24 h ↦ Exists.elim h (fun v25 h ↦ Exists.elim h (fun v26 h ↦ Exists.elim h (fun v27 h ↦ Exists.elim h (fun v28 h ↦ Exists.elim h (fun v29 h ↦ Exists.elim h (fun v30 h ↦ Exists.elim h (fun v31 h ↦ Exists.elim h (fun v32 h ↦ Exists.elim h (fun v33 h ↦ Exists.elim h (fun v34 h ↦ Exists.elim h (fun v35 h ↦ Exists.elim h (fun v36 h ↦ Exists.elim h (fun v37 h ↦ Exists.elim h (fun v38 h ↦ Exists.elim h (fun v39 h ↦ Exists.elim h (fun v40 h ↦ Exists.elim h (fun v41 h ↦ Exists.elim h (fun v42 h ↦ Exists.elim h (fun v43 h ↦ Exists.elim h (fun v44 h ↦ Exists.elim h (fun v45 h ↦ Exists.elim h (fun v46 h ↦ Exists.elim h (fun v47 h ↦ Exists.elim h (fun v48 h ↦ Exists.elim h (fun v49 h ↦ Exists.elim h (fun v50 h ↦ Exists.elim h (fun v51 h ↦ Exists.elim h (fun v52 h ↦ Exists.elim h (fun v53 h ↦ Exists.elim h (fun v54 h ↦ Exists.elim h (fun v55 h ↦ Exists.elim h (fun v56 h ↦ Exists.elim h (fun v57 h ↦ Exists.elim h (fun v58 h ↦ Exists.elim h (fun v59 h ↦ Exists.elim h (fun v60 h ↦ Exists.elim h (fun v61 h ↦ Exists.elim h (fun v62 h ↦ Exists.elim h (fun v63 h ↦ Exists.elim h (fun v64 h ↦ Exists.elim h (fun v65 h ↦ Exists.elim h (fun v66 h ↦ Exists.elim h (fun v67 h ↦ Exists.elim h (fun v68 h ↦ Exists.elim h (fun v69 h ↦ Exists.elim h (fun v70 h ↦ Exists.elim h (fun v71 h ↦ ?_))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56, a57, a58, a59, a60, a61, a62, hout⟩ := h
  let famF : Nat → Fin 2 → Fin 2 → ExactMatrix q' n 1 m' := fun k ↦
    if hk : k < L + 1 then fun a b ↦ fam4 f00 f01 f10 f11 a b ⟨k, hk⟩ else 0
  let G : Nat → (Fin 2 → Fin 2 → ExactMatrix q' n 1 m') → Fin 2 → Fin 2 → ExactMatrix q n 1 m' :=
    fun k sw ↦ branchKeyG backend base digits basis (MxxIR.selectInt (Int.ofNat k) tab).toNat
      (inA x0 x1 x2 x3 x4 x5) (lutAG hashModel key q m tag level (MxxIR.selectInt (Int.ofNat k) tab)) sw
  obtain ⟨_, rfl⟩ := familyGet_lit 0 a0
  obtain ⟨_, rfl⟩ := familyGet_lit 0 a1
  obtain ⟨_, rfl⟩ := familyGet_lit 0 a2
  obtain ⟨_, rfl⟩ := familyGet_lit 0 a3
  have hC := carried_loop_gen LoopStateG.key LoopStateG.switch famF G
    (by funext a b; simp only [famF, dif_pos (show 0 < L + 1 by omega)]; fin_cases a <;> fin_cases b <;> rfl)
    (fun k cur nxt _ hb ↦ by
      obtain ⟨hk, hk1, hs⟩ := hchild k cur nxt hb
      exact ⟨hk, by rw [hs]; simp only [famF, dif_pos hk1]⟩) a4
  have hkey0 : LoopStateG.key ((0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'),
      (0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'),
      (0 : ExactMatrix q n 1 m'), f00 ⟨0, by omega⟩, f01 ⟨0, by omega⟩, f10 ⟨0, by omega⟩,
      f11 ⟨0, by omega⟩, ()) = 0 := by
    funext a b; fin_cases a <;> fin_cases b <;> rfl
  rw [hkey0, zero_add] at hC
  obtain ⟨hLk, hLs⟩ := hC
  simp only [Int.toNat_natCast] at hLk hLs
  obtain ⟨rfl, rfl, rfl, rfl⟩ := lutA_of_samples a6 a11 a7 a12
  have hK := branchKeyG_of_stacks (c := inC x0 x1 x2 x3 x4 x5) (A := inA x0 x1 x2 x3 x4 x5)
    hn hq' layout hlayout hexact hm a16 a17 a18 a21 a22 a23 a19 a20 a24 a25 a26
    (sw := ![![v10, v11], ![v12, v13]]) (Xb := ![v36, v50]) (Fin.forall_fin_two.2 ⟨a27, a41⟩)
    (D' := ![v37, v51]) (Fin.forall_fin_two.2 ⟨a28, a42⟩)
    (D := ![v38, v52]) (Fin.forall_fin_two.2 ⟨a29, a43⟩)
    (M := ![![v39, v53], ![v41, v55]]) (Fin.forall_fin_two.2 ⟨a30, a44⟩)
    (Fin.forall_fin_two.2 ⟨a32, a46⟩) (Asw := ![![v40, v54], ![v42, v56]])
    (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨a31, a45⟩, Fin.forall_fin_two.2 ⟨a33, a47⟩⟩)
    (Z := ![v43, v57]) (Fin.forall_fin_two.2 ⟨a34, a48⟩)
    (E' := ![v44, v58]) (Fin.forall_fin_two.2 ⟨a35, a49⟩)
    (E := ![v45, v59]) (Fin.forall_fin_two.2 ⟨a36, a50⟩)
    (h := ![v14, v19]) a8 a9 a13 a14 a15
    (K := ![![v46, v60], ![v62, v64]]) (Fin.forall_fin_two.2 ⟨a37, a51⟩)
    (Fin.forall_fin_two.2 ⟨a53, a55⟩)
  have hsw3 : (![![v10, v11], ![v12, v13]] : Fin 2 → Fin 2 → ExactMatrix q' n 1 m') = famF L := hLs
  have htot : ![![v6, v7], ![v8, v9]] + ![![v46, v60], ![v62, v64]] =
      ∑ k ∈ Finset.range (L + 1), G k (famF k) := by
    rw [Finset.sum_range_succ, ← hsw3]
    exact congrArg₂ (· + ·) hLk hK
  have hsum : keyTupleG (![![v6, v7], ![v8, v9]] + ![![v46, v60], ![v62, v64]]) =
      ∑ k ∈ Finset.range (L + 1), lutKeysG backend base digits basis hashModel key tag level (sigN tab)
        (inA x0 x1 x2 x3 x4 x5) (fam4 f00 f01 f10 f11) k := by
    rw [htot, keyTupleG_sum]
    exact lut_keys_link_gen le_rfl famF (fun k hk ↦ by simp only [famF, dif_pos hk])
  rw [smudge_weight_eq] at a39
  rw [fdiv_fmod_ofNat] at hout
  subst hout
  rw [← hsum, ← rdecG_eq a39 a40, blockModSwitchRuns_eqG a38, blockModSwitchRuns_eqG a52,
    blockModSwitchRuns_eqG a54, blockModSwitchRuns_eqG a56]
  rfl

/-! ## Rows of a relation in mask form (the final combination's `hcx`) -/

/-- Row `j` of a relation is the mask part of column `j` of its keys minus the target plus a
bounded error (the `maskPart` form of `GenFinalInst.final_inst`'s row hypotheses). -/
theorem rel_row_maskPart {q m : Nat} {sm : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {t c : Fin 2 → ExactMatrix q n 1 m} {B : Nat} (h : Rel sm A t c B) (j : Fin 2) :
    ∃ e : ErrorMatrix n 1 m, CoeffBound e B ∧
      c j = Mxx.ExponentLut.Refresh.maskPart (reducePoly q n sm) (fun k ↦ A k j) - t j +
        reduceMatrix q n 1 m e := by
  obtain ⟨e, he, hc⟩ := h
  exact ⟨e j, he j, hc j⟩

/-- A selection at `i` from a list whose entries alternate `a, b, a, b, …` is `![a, b] (i mod 2)`. -/
theorem select_period2 {α : Type} {a b : α} {l : List α}
    (hl : ∀ k < l.length, l.getD k a = ![a, b] ⟨k % 2, Nat.mod_lt _ (by norm_num)⟩) {i : Nat}
    (hi : i < l.length) {out : α} (h : select (Int.ofNat i) l out) :
    out = ![a, b] ⟨i % 2, Nat.mod_lt _ (by norm_num)⟩ := by
  obtain ⟨p, hp, rfl⟩ := h
  have hpi : (p : Nat) = i := by simp only [Int.ofNat_eq_natCast] at hp; exact_mod_cast hp
  subst hpi
  rw [← hl _ hi]
  simp [List.getD_eq_getElem?_getD]

end RefreshTopProof.GenIface
