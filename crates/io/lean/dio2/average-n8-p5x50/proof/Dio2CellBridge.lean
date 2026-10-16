import Dio2Cell
import Dio2Comp
import GenKey

/-!
# The bundle bridge of the dio2 PRF cell

`PrfCellFlat.rel` destructures the computation bundle `PrfCellFlat.Comp` (variant 0, the member's
state read statically) of a dio2 PRF cell and applies `cell_flat`: the label member `idx`, whose
keys are the label's root keys of that member (`labelKeysG`), times the member's state, times the
selector right-hand side of cell `i` (tops `topPairG … (prfTopTags a b i)`).
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen

variable {n : Nat}

/-- The leading tag components of the selector tops of PRF cell `i`
(`parameters, a, program, b, member, i`); the top `j` appends `component, j`
(refresh_top's `GenBridge_Prf.prfTopTags`). -/
abbrev prfTopTags (a b : Int) (i : Nat) : List HashTagComponent :=
  [.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal a,
    .bytes [112, 114, 111, 103, 114, 97, 109], .decimal b, .bytes memberTagG, .u64Le (Int.ofNat i)]

/-- The two selector tops of a PRF cell are `topPairG` of their tags. -/
theorem prf_tops {q m : Nat} {model : HashModel} {seed : ByteArray} {pre : Blob} {a b : Int}
    {i : Nat} {t0 t1 : ExactMatrix q n 1 m}
    (h0 : hashSample model pre ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal a,
      .bytes [112, 114, 111, 103, 114, 97, 109], .decimal b, .bytes [109, 101, 109, 98, 101, 114],
      .u64Le (Int.ofNat i), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le 0]) seed t0)
    (h1 : hashSample model pre ([.bytes [112, 97, 114, 97, 109, 101, 116, 101, 114, 115], .decimal a,
      .bytes [112, 114, 111, 103, 114, 97, 109], .decimal b, .bytes [109, 101, 109, 98, 101, 114],
      .u64Le (Int.ofNat i), .bytes [99, 111, 109, 112, 111, 110, 101, 110, 116], .u64Le 1]) seed t1) :
    ![t0, t1] = topPairG model seed q m pre (prfTopTags a b i) := by
  funext j
  fin_cases j
  · exact topPairG_of_hash (pre := prfTopTags a b i) 0 (by norm_num) h0
  · exact topPairG_of_hash (pre := prfTopTags a b i) 1 (by norm_num) h1

set_option maxHeartbeats 8000000 in
/-- **A dio2 PRF cell** (online, variant 0): an encoding of `(α μ) T` with the bound of
`cell_flat`. -/
theorem PrfCellFlat.rel (hn : 0 < n) {q m m' m'' bound : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {hashModel : HashModel} {base digits base' digits' : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    {lo two width lo' two' width' lo'' two'' width'' lo''' two''' width''' : Nat}
    {pre rhsPre : Blob} {a b : Int} {i : Nat}
    {idx : Int} {x0 x1 : ExactMatrix q n 1 m} {seed seed' : ByteArray}
    {As00 As10 As01 As11 cs0 cs1 : ExactMatrix q n 1 m'} {pay : ExactMatrix q n 1 1}
    {bot0 bot1 : ExactMatrix q n 1 m''}
    {outputs : ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' ×
      ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' × Unit}
    (h : Dio2Proof.Comp.PrfCellFlat.Comp q n m' m' m' m' m'' m'' m m q q q q m'' m'' m'' m'' m''
      m'' lo two width pre lo' two' width' pre lo'' two'' width'' pre lo''' two''' width''' pre
      base digits bound m' m' m' base digits bound m' m' m' a b rhsPre base' digits' bound m''
      a b rhsPre base' digits' bound m'' m'' m'' m'' m'' backend hashModel i
      (idx, x0, seed, x1, As00, As10, cs0, pay, As01, As11, cs1, seed', bot0, bot1, ()) outputs)
    {sm sp μ α : ErrorPoly n}
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G) {Ba : Nat}
    (hα : polyNorm α ≤ 1) (hαμ : polyNorm (α * μ) ≤ 1)
    (ha : Rel sm (labelKeysG hashModel seed q m pre idx)
      (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ![x0, x1] Ba)
    (hpay : pay = fun _ _ ↦ reducePoly q n α)
    {g' : ExactMatrix q n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm ![![As00, As01], ![As10, As11]]
      (fun j ↦ reducePoly q n (μ * unitPair sp j) • g) ![cs0, cs1] Bs)
    {T : Fin 2 → ExactMatrix q n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q n sp • topPairG hashModel seed' q m'' rhsPre (prfTopTags a b i) j +
      ![bot0, bot1] j = T j + reduceMatrix q n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly q n (α * μ) • T j) ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((m' + m') * n * ((m + m) * n * Ba * bound + n * 1 * Bs) * bound + n * 1 * BB) := by
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
    v38, v39, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19,
    c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38,
    c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, h⟩ := h
  subst h
  -- The label's keys are its root keys of member `idx`.
  have hkeys : ![![v0, v4], ![v1, v5]] = labelKeysG hashModel seed q m pre idx := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 0 0 (by norm_num) c3
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 0 1 (by norm_num) c11
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 1 0 (by norm_num) c6
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 1 1 (by norm_num) c14
  rw [← hkeys] at ha
  have htop := prf_tops c35 c40
  have hrhs' : ∀ j, reducePoly q n sp • ![v26, v31] j + ![bot0, bot1] j =
      T j + reduceMatrix q n 1 m'' (eB j) := by
    rw [htop]; exact hrhs
  exact cell_flat hn hq1 layout hlayout hexact hG hα hαμ ha hpay hg' hg hs c7 c8 c15 c16 c17
    (Xs := ![v9, v17]) (fun j ↦ by fin_cases j; exacts [c18, c26])
    (D' := ![v10, v18]) (fun j ↦ by fin_cases j; exacts [c19, c27])
    (D := ![v11, v19]) (fun j ↦ by fin_cases j; exacts [c20, c28])
    (Y := ![v12, v20]) (fun j ↦ by fin_cases j; exacts [c21, c29])
    (A' := ![![v13, v21], ![v14, v22]]) (fun j ↦ by fin_cases j; exacts [c22, c30])
    (fun j ↦ by fin_cases j; exacts [c23, c31])
    (c' := ![matrixAdd v12 (matrixMulScalarRight cs0 pay),
      matrixAdd v20 (matrixMulScalarRight cs1 pay)]) (fun j ↦ by fin_cases j <;> rfl)
    c24 c25 c32 c33 c34
    (Xb := ![v27, v32]) (fun j ↦ by fin_cases j; exacts [c36, c41])
    (E' := ![v28, v33]) (fun j ↦ by fin_cases j; exacts [c37, c42])
    (E := ![v29, v34]) (fun j ↦ by fin_cases j; exacts [c38, c43])
    hrhs' heB
    (X := ![v38, v39]) (fun j ↦ by fin_cases j; exacts [c47, c48])
    (Mk := ![![v30, v35], ![v36, v37]]) (fun j ↦ by fin_cases j; exacts [c39, c44])
    (fun j ↦ by fin_cases j; exacts [c45, c46])

/-- A dynamic family read at `⌊i / 2⌋` is the member `i / 2`. -/
theorem familyGetDynamic_half {α : Type} {M : Nat} {f : Fin M → α} {i : Nat} {x : α}
    (h : familyGetDynamic f (Int.fdiv (Int.ofNat i) 2) x) (hi : i / 2 < M) : x = f ⟨i / 2, hi⟩ := by
  obtain ⟨pos, hpos, rfl⟩ := h
  congr 1
  apply Fin.ext
  have e : Int.fdiv (Int.ofNat i) 2 = ((i / 2 : Nat) : Int) := by
    rw [Int.fdiv_eq_ediv_of_nonneg _ (by norm_num), Int.ofNat_eq_natCast]; push_cast; rfl
  simp only []
  omega

set_option maxHeartbeats 8000000 in
/-- **A dio2 PRF cell** (online, variant 1): cell `i` reads the state of member `⌊i / 2⌋` from
the members' families; an encoding of `(α μ) T` with the bound of `cell_flat`. -/
theorem PrfCellFlat.rel_1 (hn : 0 < n) {q m m' m'' bound M : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {hashModel : HashModel} {base digits base' digits' : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    {lo two width lo' two' width' lo'' two'' width'' lo''' two''' width''' : Nat}
    {l0 h0 l1 h1 l2 h2 l3 h3 l4 h4 l5 h5 : Int}
    {pre rhsPre : Blob} {a b : Int} {i : Nat}
    {idx : Int} {x0 x1 : ExactMatrix q n 1 m} {seed seed' : ByteArray}
    {sA00 sA10 sA01 sA11 scs0 scs1 : Fin M → ExactMatrix q n 1 m'} {pay : ExactMatrix q n 1 1}
    {bot0 bot1 : ExactMatrix q n 1 m''}
    {outputs : ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' ×
      ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' × ExactMatrix q n 1 m'' × Unit}
    (h : Dio2Proof.Comp.PrfCellFlat.Comp_1 q n m' m' m' m' m'' m'' m m M q M M M q M M q q m'' m''
      m'' m'' m'' m'' lo two width pre lo' two' width' pre lo'' two'' width'' pre lo''' two''' width'''
      pre l0 h0 l1 h1 base digits bound m' l2 h2 m' m' l3 h3 l4 h4 base digits bound m' l5 h5 m' m'
      a b rhsPre base' digits' bound m'' a b rhsPre base' digits' bound m'' m'' m'' m'' m''
      backend hashModel i
      (idx, x0, seed, x1, sA00, sA10, scs0, pay, sA01, sA11, scs1, seed', bot0, bot1, ()) outputs)
    (hi : i / 2 < M)
    {sm sp μ α : ErrorPoly n}
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G) {Ba : Nat}
    (hα : polyNorm α ≤ 1) (hαμ : polyNorm (α * μ) ≤ 1)
    (ha : Rel sm (labelKeysG hashModel seed q m pre idx)
      (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ![x0, x1] Ba)
    (hpay : pay = fun _ _ ↦ reducePoly q n α)
    {g' : ExactMatrix q n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm ![![sA00 ⟨i / 2, hi⟩, sA01 ⟨i / 2, hi⟩], ![sA10 ⟨i / 2, hi⟩, sA11 ⟨i / 2, hi⟩]]
      (fun j ↦ reducePoly q n (μ * unitPair sp j) • g) ![scs0 ⟨i / 2, hi⟩, scs1 ⟨i / 2, hi⟩] Bs)
    {T : Fin 2 → ExactMatrix q n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q n sp • topPairG hashModel seed' q m'' rhsPre (prfTopTags a b i) j +
      ![bot0, bot1] j = T j + reduceMatrix q n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    Rel sm ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]]
      (fun j ↦ reducePoly q n (α * μ) • T j) ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]
      ((m' + m') * n * ((m + m) * n * Ba * bound + n * 1 * Bs) * bound + n * 1 * BB) := by
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
    v38, v39, v40, v41, v42, v43, v44, v45, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13,
    c14, c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32,
    c33, c34, c35, c36, c37, c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51,
    c52, c53, c54, c55, c56, c57, c58, c59, c60, c61, c62, c63, c64, c65, c66, h⟩ := h
  subst h
  have e9 := familyGetDynamic_half c20 hi
  have e10 := familyGetDynamic_half c23 hi
  have e15 := familyGetDynamic_half c30 hi
  have e20 := familyGetDynamic_half c37 hi
  have e21 := familyGetDynamic_half c40 hi
  have e26 := familyGetDynamic_half c47 hi
  subst e9 e10 e15 e20 e21 e26
  have hkeys : ![![v0, v4], ![v1, v5]] = labelKeysG hashModel seed q m pre idx := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 0 0 (by norm_num) c3
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 0 1 (by norm_num) c11
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 1 0 (by norm_num) c6
    · exact keyPairG_of_hash (pre := [.bytes memberTagG, .u64Le idx]) 1 1 (by norm_num) c14
  rw [← hkeys] at ha
  have htop := prf_tops c53 c58
  have hrhs' : ∀ j, reducePoly q n sp • ![v32, v37] j + ![bot0, bot1] j =
      T j + reduceMatrix q n 1 m'' (eB j) := by
    rw [htop]; exact hrhs
  exact cell_flat hn hq1 layout hlayout hexact hG hα hαμ ha hpay hg' hg hs c7 c8 c15 c16 c17
    (Xs := ![v11, v22]) (fun j ↦ by fin_cases j; exacts [c24, c41])
    (D' := ![v12, v23]) (fun j ↦ by fin_cases j; exacts [c25, c42])
    (D := ![v13, v24]) (fun j ↦ by fin_cases j; exacts [c26, c43])
    (Y := ![v14, v25]) (fun j ↦ by fin_cases j; exacts [c27, c44])
    (A' := ![![v16, v27], ![v17, v28]]) (fun j ↦ by fin_cases j; exacts [c31, c48])
    (fun j ↦ by fin_cases j; exacts [c32, c49])
    (c' := ![matrixAdd v14 (matrixMulScalarRight (scs0 ⟨i / 2, hi⟩) pay),
      matrixAdd v25 (matrixMulScalarRight (scs1 ⟨i / 2, hi⟩) pay)]) (fun j ↦ by fin_cases j <;> rfl)
    c33 c34 c50 c51 c52
    (Xb := ![v33, v38]) (fun j ↦ by fin_cases j; exacts [c54, c59])
    (E' := ![v34, v39]) (fun j ↦ by fin_cases j; exacts [c55, c60])
    (E := ![v35, v40]) (fun j ↦ by fin_cases j; exacts [c56, c61])
    hrhs' heB
    (X := ![v44, v45]) (fun j ↦ by fin_cases j; exacts [c65, c66])
    (Mk := ![![v36, v41], ![v42, v43]]) (fun j ↦ by fin_cases j; exacts [c57, c62])
    (fun j ↦ by fin_cases j; exacts [c63, c64])

end Dio2Proof
