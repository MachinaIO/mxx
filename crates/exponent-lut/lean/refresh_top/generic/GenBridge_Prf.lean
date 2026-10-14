import GenIface
import GenRel
import GenKey
import GenLabel

/-!
# Bundle bridges of the PRF cells

Each PRF cell role is a generated computation bundle `X.Comp` (`GenIface`): the scope relation
with its dimensions and instance constants as parameters. The bridges here destructure a bundle
and call the atom-level lemmas, so a certificate turns an adapter `<scope>.sem` into the
encoding-level conclusion of the cell.

* `PrfCell0.rel` (online, bucket 0): the private cell multiplication of `Gen.cell_mul_gen`, plus
  the cell's keys as `Gen.cellKeyG` of its input keys (the online side of the key link).
* `PrfCell1.rel` (online, buckets `≥ 1`): the same after the alignment of the input encoding to
  the lower modulus (`Gen.cell_step_aligned_gen`); the keys are `Gen.cellKeyG` of the aligned
  input keys `Gen.alignKeyG`.
* `PrfCell0.rel_1` / `PrfCell1.rel_1`: the same for a Program in its own stages (variant 1), whose
  cells hash-sample the input encoding's keys per member (`Gen.labelKeyG`) and read the payload
  directly.
* `PrfKeyCell0.rel` / `PrfKeyCell1.rel` (preprocessing): the cell's keys as the same
  `Gen.cellKeyG` of the label keys (aligned for buckets `≥ 1`) and the member's state keys, the
  tops as `Gen.topPairG`, and the selector right-hand side `s top_j + bot_j = p μ u_j G + e_j`
  (`Gen.selector_rhs_pair_gen`), which is the `hrhs`/`heB` hypothesis of the online cell.

Every bundle parameter that the atom-level lemmas need to coincide is shared in the statement
(one variable for every slot of the same gadget base, digit count, bound, basis, tag or width),
so an instance discharges them by unification; the instance-specific facts (layouts, moduli
factorizations, plaintext scale) are hypotheses. The input and output tuples are spelled as
tuples of named components (definitionally the exported tuples, by structure eta).
-/

namespace RefreshTopProof.GenIface

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen
open scoped BigOperators

variable {n : Nat}

/-- The leading tag components of the selector tops of PRF cell `i`
(`parameters, a, program, b, member, i`); the top `j` appends `component, j`. -/
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

set_option maxHeartbeats 4000000 in
/-- **PRF cell, bucket 0** (online). The input encoding `x` of `α` (payload `pay k = α`, read at
`idx = k`) is multiplied by the cell's state encoding `(As, cs)` of `p μ` and moved down by the
selector `(top, bot)`; the output encodes `p' (α μ ν)` with `cellBound`, and its keys are
`cellKeyG` of the input keys `A`, the state keys `As` and the selector. -/
theorem PrfCell0.rel (hn : 0 < n) {q q' p p' m m' m'' K bound : Nat} (hq : q = p * q')
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
    {sm sp μ ν α : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G) {Ba : Nat}
    {Kα : Nat} (hα : SmulGain α Kα) (hαμ : polyNorm (α * μ) ≤ 1)
    (ha : Rel sm ![![xA00, xA01], ![xA10, xA11]]
      (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ![x0, x1] Ba)
    (k : Fin K) (hidx : idx = ((k : Nat) : Int)) (hpay : pay k = fun _ _ ↦ reducePoly q n α)
    {g' : ExactMatrix q' n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm ![![As00, As01], ![As10, As11]]
      (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) ![cs0, cs1] Bs)
    {hT : ExactMatrix q' n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q' n sp • topPairG hashModel seed q' m'' rhsPre (prfTopTags a b i) j +
      ![bot0, bot1] j =
        reducePoly q' n ((p' : Int) * ν * unitPair sp j) • hT + reduceMatrix q' n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    EncRel sm outputs (fun j ↦ reducePoly q' n ((p' : Int) * (α * μ * ν) * unitPair sp j) • hT)
        (cellBoundK n m m' bound p Ba Bs BB Kα) ∧
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
  have hv9 : v9 = fun _ _ ↦ reducePoly q n α := by
    obtain ⟨pos, hpos, hv⟩ := c12
    have : pos = k := Fin.ext (by omega)
    subst this
    exact hv.trans hpay
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
  have hrhs' : ∀ j, reducePoly q' n sp • ![v23, v29] j + ![bot0, bot1] j =
      reducePoly q' n ((p' : Int) * ν * unitPair sp j) • hT + reduceMatrix q' n 1 m'' (eB j) := by
    rw [htop]; exact hrhs
  refine ⟨?_, ?_⟩
  · exact cell_mul_genK hn hq hq1 hq' hp layout hlayout hexact layout' hlayout' hexact' hsm hG hα hαμ
      ha hv9 hg' hg hs c1 c2 c3 c4 c5 hXs hD' hD
      (Y := ![v8, v17]) (fun j ↦ by fin_cases j; exacts [c9, c20]) hA0 hA1
      (c' := ![matrixAdd v8 (matrixMulScalarRight cs0 v9), matrixAdd v17 (matrixMulScalarRight cs1 v9)])
      (fun j ↦ by fin_cases j <;> rfl) c15 c16 c23 c24 c25 hXb hE' hE hrhs' heB
      (X := ![v39, v41]) (fun j ↦ by fin_cases j; exacts [c42, c44]) hM0 hM1
      (c'' := ![v40, v42]) (fun j ↦ by fin_cases j; exacts [c43, c45]) hA''
  · have hM := pubKeyG_of_stacks hn hq1 layout hlayout hexact hm
      (A := ![![xA00, xA01], ![xA10, xA11]]) c1 c2 c3 c4 c5 hXs hD' hD hA0 hA1
    have hK := privKeyG_of_stacks hn hq' layout' hlayout' hexact' hm'
      (c := ![matrixAdd v8 (matrixMulScalarRight cs0 v9), matrixAdd v17 (matrixMulScalarRight cs1 v9)])
      (A := ![![v10, v18], ![v11, v19]]) c15 c16 c23 c24 c25 hXb hE' hE hM0 hM1 hA''
    rw [← htop]
    exact cellKeyG_of_parts hM hK

set_option maxHeartbeats 8000000 in
/-- **PRF cell, buckets `≥ 1`** (online). The input encoding `x` of `α` at `Q` is aligned to `q`
(digits of `pA G`, a block switch), multiplied by the state encoding of member `o = ⌊i / P⌋` (of
`p μ_o`) and moved down by the selector; the output encodes `p' (α μ_o ν)`, and its keys are
`cellKeyG` of the aligned input keys `alignKeyG … (pA G)`, the member's state keys and the
selector. -/
theorem PrfCell1.rel (hn : 0 < n) {Q q q' pA p p' mA m m' m'' K W bound : Nat}
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
    {sm sp ν α : ErrorPoly n} {μ : Fin W → ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {GA : ExactMatrix Q n 1 mA} (hGA : gadgetMatrixRuns backend baseA digitsA GA) {Ba : Nat}
    {Kα : Nat} (hα : SmulGain α Kα) (hαμ : ∀ o, polyNorm (α * μ o) ≤ 1)
    (ha : Rel sm ![![xA00, xA01], ![xA10, xA11]]
      (fun j ↦ reducePoly Q n (α * unitPair sm j) • GA) ![x0, x1] Ba)
    (k : Fin K) (hidx : idx = ((k : Nat) : Int)) (hpay : pay k = fun _ _ ↦ reducePoly Q n α)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {GR : ExactMatrix Q n 1 m} (hGR : centeredRebaseRuns G GR)
    {g' : ExactMatrix q' n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : ∀ o, Rel sm ![![As00 o, As01 o], ![As10 o, As11 o]]
      (fun j ↦ reducePoly q n ((p : Int) * μ o * unitPair sp j) • g) ![cs0 o, cs1 o] Bs)
    {hT : ExactMatrix q' n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q' n sp • topPairG hashModel seed q' m'' rhsPre (prfTopTags a b i) j +
      ![bot0, bot1] j =
        reducePoly q' n ((p' : Int) * ν * unitPair sp j) • hT + reduceMatrix q' n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    ∃ o : Fin W, ((o : Nat) : Int) = Int.fdiv (Int.ofNat i) P ∧
      EncRel sm outputs
        (fun j ↦ reducePoly q' n ((p' : Int) * (α * μ o * ν) * unitPair sp j) • hT)
        (cellBoundK n m m' bound p ((mA * n * Ba * (baseA.toNat / 2) + (n + 2) * (pA / 2)) / pA)
          Bs BB Kα) ∧
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
  have hrhs' : ∀ j, reducePoly q' n sp • ![v42, v48] j + ![bot0, bot1] j =
      reducePoly q' n ((p' : Int) * ν * unitPair sp j) • hT + reduceMatrix q' n 1 m'' (eB j) := by
    rw [htop]; exact hrhs
  refine ⟨o, ho, ?_, ?_⟩
  · exact cell_step_aligned_genK hn hQ hQ1 hpA hq hq1 hq' hp layoutA hlayoutA hexactA layout hlayout
      hexact layout' hlayout' hexact' hsm hGA (ca0 := fun _ ↦ x0) (ca1 := fun _ ↦ x1)
      (aA00 := fun _ ↦ xA00) (aA10 := fun _ ↦ xA10) (aA01 := fun _ ↦ xA01) (aA11 := fun _ ↦ xA11)
      (apay := fun _ ↦ pay k) (α := fun _ ↦ α) (fun _ ↦ hα) (fun _ ↦ hαμ o) (fun _ ↦ ha)
      (fun _ ↦ hpay) k hidx ⟨k, hidx.symm, rfl⟩ ⟨k, hidx.symm, rfl⟩ ⟨k, hidx.symm, rfl⟩
      ⟨k, hidx.symm, rfl⟩ ⟨k, hidx.symm, rfl⟩ ⟨k, hidx.symm, rfl⟩ ⟨k, hidx.symm, rfl⟩
      hG hGR hscaled c3 c4 c12 c8 c9 c13 c14 c34 hg' hg (hs o) c10 c11 c15 c16 c17 hXs hD' hD
      (Y := ![v22, v35]) (fun j ↦ by fin_cases j; exacts [c27, c48]) hA0 hA1
      (c' := ![matrixAdd v22 (matrixMulScalarRight (cs0 o) v25),
        matrixAdd v35 (matrixMulScalarRight (cs1 o) v25)])
      (fun j ↦ by fin_cases j <;> rfl) c37 c38 c54 c55 c56 hXb hE' hE hrhs' heB
      (X := ![v58, v60]) (fun j ↦ by fin_cases j; exacts [c73, c75]) hM0 hM1
      (c'' := ![v59, v61]) (fun j ↦ by fin_cases j; exacts [c74, c76]) hA''
  · have hal := alignKeyG_of_runs (A := ![![xA00, xA01], ![xA10, xA11]])
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

set_option maxHeartbeats 4000000 in
/-- **PRF cell, bucket 0, of a Program in its own stages** (online; `prf.cell0`, variant 1): as
`PrfCell0.rel`, with the input encoding's keys hash-sampled from the key at the gathered member
`idx` (`labelKeyG`), the payload read directly, and the key as the selector tops' seed. -/
theorem PrfCell0.rel_1 (hn : 0 < n) {q q' p p' m m' m'' bound E : Nat} (hq : q = p * q')
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
      (idx, x0, key, x1, As00, As10, cs0, pay, As01, As11, cs1, bot0, bot1, ()) outputs)
    {sm sp μ ν α : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G) {Ba : Nat}
    {Kα : Nat} (hα : SmulGain α Kα) (hαμ : polyNorm (α * μ) ≤ 1)
    (ha : Rel sm ![![labelKeyG hashModel q n m key pre idx 0, labelKeyG hashModel q n m key pre idx 1], ![labelKeyG hashModel q n m key pre idx 2, labelKeyG hashModel q n m key pre idx 3]]
      (fun j ↦ reducePoly q n (α * unitPair sm j) • G) ![x0, x1] Ba)
    (hpay : pay = fun _ _ ↦ reducePoly q n α)
    {g' : ExactMatrix q' n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : Rel sm ![![As00, As01], ![As10, As11]]
      (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) ![cs0, cs1] Bs)
    {hT : ExactMatrix q' n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q' n sp • topPairG hashModel key q' m'' rhsPre (prfTopTags a b i) j +
      ![bot0, bot1] j =
        reducePoly q' n ((p' : Int) * ν * unitPair sp j) • hT + reduceMatrix q' n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    EncRel sm outputs (fun j ↦ reducePoly q' n ((p' : Int) * (α * μ * ν) * unitPair sp j) • hT)
        (cellBoundK n m m' bound p Ba Bs BB Kα) ∧
      Enc.A outputs = cellKeyG backend base digits base' digits' basis
        ![![labelKeyG hashModel q n m key pre idx 0, labelKeyG hashModel q n m key pre idx 1], ![labelKeyG hashModel q n m key pre idx 2, labelKeyG hashModel q n m key pre idx 3]] ![![As00, As01], ![As10, As11]]
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
  have hrhs' : ∀ j, reducePoly q' n sp • ![v26, v32] j + ![bot0, bot1] j =
      reducePoly q' n ((p' : Int) * ν * unitPair sp j) • hT + reduceMatrix q' n 1 m'' (eB j) := by
    rw [htop]; exact hrhs
  refine ⟨?_, ?_⟩
  · exact cell_mul_genK hn hq hq1 hq' hp layout hlayout hexact layout' hlayout' hexact' hsm hG hα hαμ
      ha hpay hg' hg hs c7 c8 c15 c16 c17 hXs hD' hD
      (Y := ![v12, v20]) (fun j ↦ by fin_cases j; exacts [c21, c29]) hA0 hA1
      (c' := ![matrixAdd v12 (matrixMulScalarRight cs0 pay), matrixAdd v20 (matrixMulScalarRight cs1 pay)])
      (fun j ↦ by fin_cases j <;> rfl) c24 c25 c32 c33 c34 hXb hE' hE hrhs' heB
      (X := ![v42, v44]) (fun j ↦ by fin_cases j; exacts [c51, c53]) hM0 hM1
      (c'' := ![v43, v45]) (fun j ↦ by fin_cases j; exacts [c52, c54]) hA''
  · have hM := pubKeyG_of_stacks hn hq1 layout hlayout hexact hm
      (A := ![![labelKeyG hashModel q n m key pre idx 0, labelKeyG hashModel q n m key pre idx 1], ![labelKeyG hashModel q n m key pre idx 2, labelKeyG hashModel q n m key pre idx 3]]) c7 c8 c15 c16 c17 hXs hD' hD hA0 hA1
    have hK := privKeyG_of_stacks hn hq' layout' hlayout' hexact' hm'
      (c := ![matrixAdd v12 (matrixMulScalarRight cs0 pay), matrixAdd v20 (matrixMulScalarRight cs1 pay)])
      (A := ![![v13, v21], ![v14, v22]]) c24 c25 c32 c33 c34 hXb hE' hE hM0 hM1 hA''
    rw [← htop]
    exact cellKeyG_of_parts hM hK

set_option maxHeartbeats 8000000 in
/-- **PRF cell, buckets `≥ 1`, of a Program in its own stages** (online; `prf.cell1`, variant 1):
as `PrfCell1.rel`, with the input encoding's keys hash-sampled from the key at the gathered member
`idx` (`labelKeyG`), the payload read directly, and the key as the selector tops' seed. -/
theorem PrfCell1.rel_1 (hn : 0 < n) {Q q q' pA p p' mA m m' m'' K W bound E : Nat}
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
    {sm sp ν α : ErrorPoly n} {μ : Fin W → ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {GA : ExactMatrix Q n 1 mA} (hGA : gadgetMatrixRuns backend baseA digitsA GA) {Ba : Nat}
    {Kα : Nat} (hα : SmulGain α Kα) (hαμ : ∀ o, polyNorm (α * μ o) ≤ 1)
    (ha : Rel sm ![![labelKeyG hashModel Q n mA key pre idx 0, labelKeyG hashModel Q n mA key pre idx 1], ![labelKeyG hashModel Q n mA key pre idx 2, labelKeyG hashModel Q n mA key pre idx 3]]
      (fun j ↦ reducePoly Q n (α * unitPair sm j) • GA) ![x0, x1] Ba)
    (k : Fin K) (hidx : idx = ((k : Nat) : Int)) (hpay : pay = fun _ _ ↦ reducePoly Q n α)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {GR : ExactMatrix Q n 1 m} (hGR : centeredRebaseRuns G GR)
    {g' : ExactMatrix q' n 1 m'} (hg' : gadgetMatrixRuns backend base' digits' g')
    {g : ExactMatrix q n 1 m'} (hg : centeredRebaseRuns g' g) {Bs : Nat}
    (hs : ∀ o, Rel sm ![![As00 o, As01 o], ![As10 o, As11 o]]
      (fun j ↦ reducePoly q n ((p : Int) * μ o * unitPair sp j) • g) ![cs0 o, cs1 o] Bs)
    {hT : ExactMatrix q' n 1 m''} {eB : Fin 2 → ErrorMatrix n 1 m''} {BB : Nat}
    (hrhs : ∀ j, reducePoly q' n sp • topPairG hashModel key q' m'' rhsPre (prfTopTags a b i) j +
      ![bot0, bot1] j =
        reducePoly q' n ((p' : Int) * ν * unitPair sp j) • hT + reduceMatrix q' n 1 m'' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) :
    ∃ o : Fin W, ((o : Nat) : Int) = Int.fdiv (Int.ofNat i) P ∧
      EncRel sm outputs
        (fun j ↦ reducePoly q' n ((p' : Int) * (α * μ o * ν) * unitPair sp j) • hT)
        (cellBoundK n m m' bound p ((mA * n * Ba * (baseA.toNat / 2) + (n + 2) * (pA / 2)) / pA)
          Bs BB Kα) ∧
      Enc.A outputs = cellKeyG backend base digits base' digits' basis
        (alignKeyG backend baseA digitsA basisA ![![labelKeyG hashModel Q n mA key pre idx 0, labelKeyG hashModel Q n mA key pre idx 1], ![labelKeyG hashModel Q n mA key pre idx 2, labelKeyG hashModel Q n mA key pre idx 3]]
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
  have hrhs' : ∀ j, reducePoly q' n sp • ![v45, v51] j + ![bot0, bot1] j =
      reducePoly q' n ((p' : Int) * ν * unitPair sp j) • hT + reduceMatrix q' n 1 m'' (eB j) := by
    rw [htop]; exact hrhs
  refine ⟨o, ho, ?_, ?_⟩
  · exact cell_step_aligned_genK hn hQ hQ1 hpA hq hq1 hq' hp layoutA hlayoutA hexactA layout hlayout
      hexact layout' hlayout' hexact' hsm hGA (ca0 := fun _ ↦ x0) (ca1 := fun _ ↦ x1)
      (aA00 := fun _ ↦ labelKeyG hashModel Q n mA key pre idx 0) (aA10 := fun _ ↦ labelKeyG hashModel Q n mA key pre idx 2) (aA01 := fun _ ↦ labelKeyG hashModel Q n mA key pre idx 1) (aA11 := fun _ ↦ labelKeyG hashModel Q n mA key pre idx 3)
      (apay := fun _ ↦ pay) (α := fun _ ↦ α) (fun _ ↦ hα) (fun _ ↦ hαμ o) (fun _ ↦ ha)
      (fun _ ↦ hpay) k hidx ⟨k, hidx.symm, rfl⟩ ⟨k, hidx.symm, rfl⟩ ⟨k, hidx.symm, rfl⟩
      ⟨k, hidx.symm, rfl⟩ ⟨k, hidx.symm, rfl⟩ ⟨k, hidx.symm, rfl⟩ ⟨k, hidx.symm, rfl⟩
      hG hGR hscaled c3 c4 c18 c11 c15 c22 c26 c43 hg' hg (hs o) c16 c17 c27 c28 c29 hXs hD' hD
      (Y := ![v26, v38]) (fun j ↦ by fin_cases j; exacts [c39, c57]) hA0 hA1
      (c' := ![matrixAdd v26 (matrixMulScalarRight (cs0 o) v28),
        matrixAdd v38 (matrixMulScalarRight (cs1 o) v28)])
      (fun j ↦ by fin_cases j <;> rfl) c46 c47 c63 c64 c65 hXb hE' hE hrhs' heB
      (X := ![v61, v63]) (fun j ↦ by fin_cases j; exacts [c82, c84]) hM0 hM1
      (c'' := ![v62, v64]) (fun j ↦ by fin_cases j; exacts [c83, c85]) hA''
  · have hal := alignKeyG_of_runs (A := ![![labelKeyG hashModel Q n mA key pre idx 0, labelKeyG hashModel Q n mA key pre idx 1], ![labelKeyG hashModel Q n mA key pre idx 2, labelKeyG hashModel Q n mA key pre idx 3]])
      (A' := ![![v8, v15], ![v10, v17]]) (basis := basisA) c3
      (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [c11, c22, c15, c26])
    rw [hscaled] at hal
    have hM := pubKeyG_of_stacks hn hq1 layout hlayout hexact hm
      (A := ![![v8, v15], ![v10, v17]]) c16 c17 c27 c28 c29 hXs hD' hD hA0 hA1
    have hK := privKeyG_of_stacks hn hq' layout' hlayout' hexact' hm'
      (c := ![matrixAdd v26 (matrixMulScalarRight (cs0 o) v28),
        matrixAdd v38 (matrixMulScalarRight (cs1 o) v28)])
      (A := ![![v29, v40], ![v30, v41]]) c46 c47 c63 c64 c65 hXb hE' hE hM0 hM1 hA''
    rw [← htop, ← hal]
    exact cellKeyG_of_parts hM hK

set_option maxHeartbeats 8000000 in
/-- **PRF cell keys, bucket 0** (preprocessing). The keys are `cellKeyG` of the label keys of
member `mem` and the member's state keys `As`; the tops are `topPairG`; the bottoms satisfy the
selector right-hand side `s top_j + bot_j = p μ u_j G + e_j` (`μ = c X^e`, `u = (s, 1)`). -/
theorem PrfKeyCell0.rel (hn : 0 < n) {q q' qG m m' m'' bound' : Nat} (hq1 : 1 < q) (hq' : 1 < q')
    {backend : BackendContext} {hashModel : HashModel}
    {base digits base' digits' gbase gdigits : Int}
    {lo rb : Int} {re : Nat} {labelPre rhsPre : Blob} {a b pp cutoff : Int} {σ : Rat}
    {basis : List Nat} {i : Nat}
    {seed : ByteArray} {mem c e : Int} {As00 As10 As01 As11 : ExactMatrix q n 1 m'}
    {sec one : ExactMatrix q n 1 1} {K00 K01 K10 K11 t0 t1 b0 b1 : ExactMatrix q' n 1 m''}
    (h : PrfKeyCell0.Comp q n m' m' q q' m'' q q q' m'' q' m'' q' m'' q' m'' m m m' m' qG
      lo rb re labelPre lo rb re labelPre base digits lo rb re labelPre lo rb re labelPre
      base digits m' m' a b rhsPre base' digits' bound' base digits base digits m' m'
      gbase gdigits σ cutoff base' digits' [pp] bound' m'' basis a b rhsPre base' digits' bound'
      σ cutoff base' digits' [pp] bound' m'' basis m'' basis m'' basis [pp] [pp]
      backend hashModel i (seed, mem, As00, As10, As01, As11, sec, c, e, one, ())
      (K00, K01, K10, K11, t0, t1, b0, b1, ()))
    {sp : ErrorPoly n} (hsec : sec = fun _ _ ↦ reducePoly q n sp) (hone : one = 1)
    {G' : ExactMatrix qG n 1 m''} (hG' : gadgetMatrixRuns backend gbase gdigits G')
    {G : ExactMatrix q' n 1 m''} (hG : centeredRebaseRuns G' G) :
    ![![K00, K01], ![K10, K11]] = cellKeyG backend base digits base' digits' basis
        (labelKeysG hashModel seed q m labelPre mem) ![![As00, As01], ![As10, As11]]
        (topPairG hashModel seed q' m'' rhsPre (prfTopTags a b i)) ![b0, b1] ∧
      ![t0, t1] = topPairG hashModel seed q' m'' rhsPre (prfTopTags a b i) ∧
      ∃ eB : Fin 2 → ErrorMatrix n 1 m'', (∀ j, CoeffBound (eB j) cutoff.toNat) ∧ ∀ j,
        reducePoly q' n sp • topPairG hashModel seed q' m'' rhsPre (prfTopTags a b i) j +
          ![b0, b1] j =
        reducePoly q' n ((pp : ErrorPoly n) * message c e * unitPair sp j) • G +
          reduceMatrix q' n 1 m'' (eB j) := by
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
  have c50 := h
  simp only [Prod.mk.injEq] at c50
  obtain ⟨hK00, hK01, hK10, hK11, ht0, ht1, hb0, hb1, -⟩ := c50
  subst K00 K01 K10 K11 t0 t1
  rw [← hb0] at c33
  rw [← hb1] at c42
  have htop := prf_tops c20 c37
  have hM := pubKeyG_of_pre (A := labelKeysG hashModel seed q m labelPre mem)
    (As := ![![As00, As01], ![As10, As11]]) (d0 := ![v3, v14]) (d1 := ![v7, v15])
    (M := ![![v8, v16], ![v9, v17]])
    (by
      rw [show labelKeysG hashModel seed q m labelPre mem 0 0 = v0 from
          (keyPairG_of_hash 0 0 (by norm_num) c3).symm,
        show labelKeysG hashModel seed q m labelPre mem 1 0 = v1 from
          (keyPairG_of_hash 1 0 (by norm_num) c6).symm]
      exact c7)
    (by
      rw [show labelKeysG hashModel seed q m labelPre mem 0 1 = v4 from
          (keyPairG_of_hash 0 1 (by norm_num) c11).symm,
        show labelKeysG hashModel seed q m labelPre mem 1 1 = v5 from
          (keyPairG_of_hash 1 1 (by norm_num) c14).symm]
      exact c15)
    (fun j ↦ by fin_cases j; exacts [c8, c23]) (fun j ↦ by fin_cases j; exacts [c16, c24])
    (fun j ↦ by fin_cases j; exacts [c17, c25]) (fun j ↦ by fin_cases j; exacts [c18, c26])
  have hK := privKeyG_of_pre (A := ![![v8, v16], ![v9, v17]])
    (top := ![v11, v28]) (bot := ![b0, b1])
    (dT := ![v12, v29]) (rT := ![v13, v30]) (dB := ![v24, v33]) (rB := ![v25, v34])
    (M := ![![v26, v35], ![v37, v39]]) (basis := basis)
    (A' := ![![v27, v36], ![v38, v40]]) c19 c27
    (fun j ↦ by fin_cases j; exacts [c21, c38]) (fun j ↦ by fin_cases j; exacts [c22, c39])
    (fun j ↦ by fin_cases j; exacts [c33, c42]) (fun j ↦ by fin_cases j; exacts [c34, c43])
    (fun j ↦ by fin_cases j; exacts [c35, c44]) (fun j ↦ by fin_cases j; exacts [c46, c48])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [c36, c45, c47, c49])
  have hs : v19 = fun _ _ ↦ reducePoly q' n sp := by
    rw [hsec] at c28; exact modulusReduce_poly hq1 hq' hn c28
  have hone' : v31 = 1 := by
    rw [hone] at c40; exact modulusReduce_one hq1 hq' hn c40
  have hmsg : v22 = fun _ _ ↦ reducePoly q' n (message c e) := by
    rw [multiplyMonomial_liftInteger] at c31; exact modulusReduce_poly hq1 hq' hn c31
  have hGv : v21 = G := by
    rw [gadgetMatrixRuns_unique c29 hG'] at c30
    exact centeredRebaseRuns_unique c30 hG
  obtain ⟨eB, heB, hrhs⟩ := selector_rhs_pair_gen (top := ![v11, v28]) (bot := ![b0, b1])
    (e := ![v23, v32]) hs hone' hmsg (fun j ↦ by fin_cases j; exacts [c32, c41]) hb0 hb1
  rw [← htop, hGv] at *
  exact ⟨cellKeyG_of_parts hM hK, rfl, eB, heB, hrhs⟩

set_option maxHeartbeats 8000000 in
/-- **PRF cell keys, buckets `≥ 1`** (preprocessing). As `PrfKeyCell0.rel`, with the label keys
aligned to `q` (`alignKeyG … (pA G)`) and the state keys of member `o = ⌊i / P⌋`. -/
theorem PrfKeyCell1.rel (hn : 0 < n) {Q q q' qG pA mL m m' m'' W bound' : Nat} (hQ1 : 1 < Q)
    (hq' : 1 < q')
    {backend : BackendContext} {hashModel : HashModel}
    {baseA digitsA base digits base' digits' gbase gdigits : Int}
    {lo rb lo2 hi2 P : Int} {re : Nat} {labelPre rhsPre : Blob} {a b pp cutoff pAi : Int}
    (hpAi : (pA : Int) = pAi) {σ : Rat} {basisA basis : List Nat} {i : Nat}
    {seed : ByteArray} {mem c e : Int} {As00 As10 As01 As11 : Fin W → ExactMatrix q n 1 m'}
    {sec one : ExactMatrix Q n 1 1} {K00 K01 K10 K11 t0 t1 b0 b1 : ExactMatrix q' n 1 m''}
    (h : PrfKeyCell1.Comp Q n m q m' m' Q q' m'' W W W W Q Q q' m'' q' m'' q' m'' q' m'' mL q m' m'
      qG
      lo rb re labelPre base digits baseA digitsA [pAi] basisA lo rb re labelPre basisA
      lo2 P hi2 P P base digits lo rb re labelPre basisA lo rb re labelPre basisA
      lo2 P hi2 P P base digits m' m' a b rhsPre base' digits' bound' lo2 P hi2 P P base digits
      lo2 P hi2 P P base digits m' m' gbase gdigits σ cutoff base' digits' [pp] bound' m'' basis
      a b rhsPre base' digits' bound' σ cutoff base' digits' [pp] bound' m'' basis m'' basis
      m'' basis [pp] [pp]
      backend hashModel i (seed, mem, As00, As10, As01, As11, sec, c, e, one, ())
      (K00, K01, K10, K11, t0, t1, b0, b1, ()))
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {GR : ExactMatrix Q n 1 m} (hGR : centeredRebaseRuns G GR)
    {sp : ErrorPoly n} (hsec : sec = fun _ _ ↦ reducePoly Q n sp) (hone : one = 1)
    {GT' : ExactMatrix qG n 1 m''} (hGT' : gadgetMatrixRuns backend gbase gdigits GT')
    {GT : ExactMatrix q' n 1 m''} (hGT : centeredRebaseRuns GT' GT) :
    ∃ o : Fin W, ((o : Nat) : Int) = Int.fdiv (Int.ofNat i) P ∧
      ![![K00, K01], ![K10, K11]] = cellKeyG backend base digits base' digits' basis
        (alignKeyG backend baseA digitsA basisA (labelKeysG hashModel seed Q mL labelPre mem)
          (reducePoly Q n ((pA : Int) : ErrorPoly n) • GR))
        ![![As00 o, As01 o], ![As10 o, As11 o]]
        (topPairG hashModel seed q' m'' rhsPre (prfTopTags a b i)) ![b0, b1] ∧
      ![t0, t1] = topPairG hashModel seed q' m'' rhsPre (prfTopTags a b i) ∧
      ∃ eB : Fin 2 → ErrorMatrix n 1 m'', (∀ j, CoeffBound (eB j) cutoff.toNat) ∧ ∀ j,
        reducePoly q' n sp • topPairG hashModel seed q' m'' rhsPre (prfTopTags a b i) j +
          ![b0, b1] j =
        reducePoly q' n ((pp : ErrorPoly n) * message c e * unitPair sp j) • GT +
          reduceMatrix q' n 1 m'' (eB j) := by
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
  have c69 := h
  simp only [Prod.mk.injEq] at c69
  obtain ⟨hK00, hK01, hK10, hK11, ht0, ht1, hb0, hb1, -⟩ := c69
  subst K00 K01 K10 K11 t0 t1
  rw [← hb0] at c52
  rw [← hb1] at c61
  have e1 := gadgetMatrixRuns_unique c4 hG
  subst v1
  have e2 := centeredRebaseRuns_unique c5 hGR
  subst v2
  obtain ⟨o, ho, rfl⟩ := c15
  obtain ⟨o1, ho1, rfl⟩ := c28
  obtain ⟨o2, ho2, rfl⟩ := c38
  obtain ⟨o3, ho3, rfl⟩ := c42
  have e1o : o1 = o := Fin.ext (by have := ho1.trans ho.symm; omega)
  have e2o : o2 = o := Fin.ext (by have := ho2.trans ho.symm; omega)
  have e3o : o3 = o := Fin.ext (by have := ho3.trans ho.symm; omega)
  subst o1 o2 o3
  have hscaled : matrixMulScalarLeft (matrixPolynomial [((pA : Nat) : Int)]) GR =
      reducePoly Q n ((pA : Int) : ErrorPoly n) • GR := by
    rw [matrixMulScalarLeft_eq, matrixPolynomial_reduceG]
  have htop := prf_tops c33 c56
  have hL : ∀ k j, labelKeysG hashModel seed Q mL labelPre mem k j = ![![v0, v10], ![v5, v12]] k j := by
    intro k j
    fin_cases k <;> fin_cases j
    · exact (keyPairG_of_hash 0 0 (by norm_num) c3).symm
    · exact (keyPairG_of_hash 0 1 (by norm_num) c19).symm
    · exact (keyPairG_of_hash 1 0 (by norm_num) c10).symm
    · exact (keyPairG_of_hash 1 1 (by norm_num) c23).symm
  have hAl := alignKeyG_of_runs (A := labelKeysG hashModel seed Q mL labelPre mem) (q := q)
    (A' := ![![v4, v11], ![v6, v13]]) (basis := basisA) c6
    (fun k j ↦ by
      rw [hL k j]
      fin_cases k <;> fin_cases j; exacts [c7, c20, c11, c24])
  rw [hscaled] at hAl
  have hM := pubKeyG_of_pre (A := ![![v4, v11], ![v6, v13]])
    (As := ![![As00 o, As01 o], ![As10 o, As11 o]]) (d0 := ![v9, v24]) (d1 := ![v16, v26])
    (M := ![![v17, v27], ![v18, v28]]) c12 c25
    (fun j ↦ by fin_cases j; exacts [c16, c39]) (fun j ↦ by fin_cases j; exacts [c29, c43])
    (fun j ↦ by fin_cases j; exacts [c30, c44]) (fun j ↦ by fin_cases j; exacts [c31, c45])
  have hK := privKeyG_of_pre (A := ![![v17, v27], ![v18, v28]])
    (top := ![v20, v39]) (bot := ![b0, b1])
    (dT := ![v21, v40]) (rT := ![v22, v41]) (dB := ![v35, v44]) (rB := ![v36, v45])
    (M := ![![v37, v46], ![v48, v50]]) (basis := basis)
    (A' := ![![v38, v47], ![v49, v51]]) c32 c46
    (fun j ↦ by fin_cases j; exacts [c34, c57]) (fun j ↦ by fin_cases j; exacts [c35, c58])
    (fun j ↦ by fin_cases j; exacts [c52, c61]) (fun j ↦ by fin_cases j; exacts [c53, c62])
    (fun j ↦ by fin_cases j; exacts [c54, c63]) (fun j ↦ by fin_cases j; exacts [c65, c67])
    (fun k j ↦ by fin_cases k <;> fin_cases j; exacts [c55, c64, c66, c68])
  have hs : v30 = fun _ _ ↦ reducePoly q' n sp := by
    rw [hsec] at c47; exact modulusReduce_poly hQ1 hq' hn c47
  have hone' : v42 = 1 := by
    rw [hone] at c59; exact modulusReduce_one hQ1 hq' hn c59
  have hmsg : v33 = fun _ _ ↦ reducePoly q' n (message c e) := by
    rw [multiplyMonomial_liftInteger] at c50; exact modulusReduce_poly hQ1 hq' hn c50
  have hGv : v32 = GT := by
    rw [gadgetMatrixRuns_unique c48 hGT'] at c49
    exact centeredRebaseRuns_unique c49 hGT
  obtain ⟨eB, heB, hrhs⟩ := selector_rhs_pair_gen (top := ![v20, v39]) (bot := ![b0, b1])
    (e := ![v34, v43]) hs hone' hmsg (fun j ↦ by fin_cases j; exacts [c51, c60]) hb0 hb1
  rw [← htop, hGv] at *
  refine ⟨o, ho, ?_, rfl, eB, heB, hrhs⟩
  rw [← hAl]
  exact cellKeyG_of_parts hM hK

end RefreshTopProof.GenIface
