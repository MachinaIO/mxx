import Dio2Keys
import GenInstLemmas

/-!
# The keys of dio2 bundles and loops

Key forms (no semantic hypotheses) of the remaining dio2 computation bundles, completing
`Dio2Keys`: the dynamic PRF cell, the row sum of a bucket, the full LUT member loop
(`LutMemberFlat`), the plain-monomial and `fmod` placements, the member and column totals, and the
stand-alone LUT loop step (`LutRoundStep.Comp_1`). Each states `keys of the outputs = F (input keys,
hashed root keys, switching artifacts)` with `F` an explicit function, so the online stage and the
keys stage (the same bundles on zero rows and other payloads) compute equal keys from equal inputs.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen RefreshTopProof.GenIface
  RefreshTopProof.GenInst

variable {n : Nat}

/-! ## The dynamic PRF cell -/

set_option maxHeartbeats 8000000 in
/-- **The keys of a dio2 PRF cell** (variant 1, the state of member `⌊i / 2⌋` read from the
members' families): `cellKeyF` of the label member's root keys, the state keys of member `i / 2`
and the selector RHS — whatever the rows and the payload. -/
theorem PrfCellFlat.keys_1 (hn : 0 < n) {q m m' m'' bound M : Nat} (hq1 : 1 < q)
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
    (hi : i / 2 < M) :
    ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] =
      cellKeyF backend base digits base' digits' (labelKeysG hashModel seed q m pre idx)
        ![![sA00 ⟨i / 2, hi⟩, sA01 ⟨i / 2, hi⟩], ![sA10 ⟨i / 2, hi⟩, sA11 ⟨i / 2, hi⟩]]
        (topPairG hashModel seed' q m'' rhsPre (prfTopTags a b i)) ![bot0, bot1] := by
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
  have htop := prf_tops c53 c58
  have hm := width_of_stack hlayout c25
  have hpub := pubKeyG_of_stacks hn hq1 layout hlayout hexact hm
    (A := ![![v0, v4], ![v1, v5]]) c7 c8 c15 c16 c17
    (As := ![![sA00 ⟨i / 2, hi⟩, sA01 ⟨i / 2, hi⟩], ![sA10 ⟨i / 2, hi⟩, sA11 ⟨i / 2, hi⟩]])
    (Xs := ![v11, v22]) (fun j ↦ by fin_cases j; exacts [c24, c41])
    (D' := ![v12, v23]) (fun j ↦ by fin_cases j; exacts [c25, c42])
    (D := ![v13, v24]) (fun j ↦ by fin_cases j; exacts [c26, c43])
    (M := ![![v16, v27], ![v17, v28]]) (fun j ↦ by fin_cases j; exacts [c31, c48])
    (fun j ↦ by fin_cases j; exacts [c32, c49])
  have hpriv := privKeyF_of_stacks hn hq1 layout hlayout hexact
    (c := ![matrixAdd v14 (matrixMulScalarRight (scs0 ⟨i / 2, hi⟩) pay),
      matrixAdd v25 (matrixMulScalarRight (scs1 ⟨i / 2, hi⟩) pay)])
    (A := ![![v16, v27], ![v17, v28]]) c33 c34 c50 c51 c52
    (top := ![v32, v37]) (bot := ![bot0, bot1])
    (Xb := ![v33, v38]) (fun j ↦ by fin_cases j; exacts [c54, c59])
    (D' := ![v34, v39]) (fun j ↦ by fin_cases j; exacts [c55, c60])
    (D := ![v35, v40]) (fun j ↦ by fin_cases j; exacts [c56, c61])
    (M := ![![v36, v41], ![v42, v43]]) (fun j ↦ by fin_cases j; exacts [c57, c62])
    (fun j ↦ by fin_cases j; exacts [c63, c64])
  rw [cellKeyF, ← hkeys, ← htop, ← hpub, ← hpriv]


/-! ## Row sums, placements and totals -/

/-- The keys of member `o` of a bucket whose cells have keys `cell`: the sum of its `W` cells'
keys. -/
noncomputable def rowKeyF {q m M W : Nat} (cell : Fin (M * W) → Fin 2 → Fin 2 → ExactMatrix q n 1 m)
    (o : Fin M) : Fin 2 → Fin 2 → ExactMatrix q n 1 m :=
  ∑ j : Fin W, cell (cellAt o j)

/-- **The keys of a row sum** (`prf.rowsum`): the sum of the member's cells' keys. -/
theorem rowsum_keys {q m M W : Nat} {fams : Tup (Fin (M * W) → ExactMatrix q n 1 m) 6} {o : Fin M}
    {out : Tup (ExactMatrix q n 1 m) 6} (h : RowSum.Spec fams W o out) :
    Enc.A out = rowKeyF (fun i ↦ Enc.A (Tup.ev fams i)) o :=
  enc_sum_A _ out fun c ↦ by rw [rowSum_get h c]; simp

/-- The keys of a bucket row from its cells' key forms `K` (e.g. `cellKeyF …` of
`PrfCellFlat.keys` / `PrfCellFlat.keys_1`). -/
theorem bucket_keys {q m M W : Nat} {fams : Tup (Fin (M * W) → ExactMatrix q n 1 m) 6} {o : Fin M}
    {out : Tup (ExactMatrix q n 1 m) 6} (h : RowSum.Spec fams W o out)
    {K : Fin (M * W) → Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    (hcell : ∀ j : Fin W, Enc.A (Tup.ev fams (cellAt o j)) = K (cellAt o j)) :
    Enc.A out = rowKeyF K o := by
  rw [rowsum_keys h, rowKeyF, rowKeyF]
  exact Finset.sum_congr rfl fun j _ ↦ hcell j

/-- The keys multiplied by the monomial `X^e`. -/
noncomputable def monoKeyF {q m : Nat} (e : Int) (A : Fin 2 → Fin 2 → ExactMatrix q n 1 m) :
    Fin 2 → Fin 2 → ExactMatrix q n 1 m :=
  fun a j ↦ multiplyMonomial (A a j) e

/-- **The keys of a placement** (`sampler.placement`): the input keys times `X^(o mod N)`. -/
theorem placement_keys {q m : Nat} {N : Int} {o : Nat} {inp out : Tup (ExactMatrix q n 1 m) 6}
    (h : Placement.Spec N o inp out) :
    Enc.A out = monoKeyF (Int.fmod (o : Int) N) (Enc.A inp) := by
  funext a j
  fin_cases a <;> fin_cases j <;> simp [Enc.A, monoKeyF, h _]

/-- **The keys of a plain-monomial placement** (`sampler.monomial`): the input keys times
`X^o`. -/
theorem monomial_keys {q m : Nat} {o : Nat} {x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m}
    {out : Tup (ExactMatrix q n 1 m) 6}
    (h : out = (multiplyMonomial x0 (Int.ofNat o), (multiplyMonomial x1 (Int.ofNat o),
      (multiplyMonomial x2 (Int.ofNat o), (multiplyMonomial x3 (Int.ofNat o),
        (multiplyMonomial x4 (Int.ofNat o), (multiplyMonomial x5 (Int.ofNat o), ()))))))) :
    Enc.A out = monoKeyF (Int.ofNat o) (Enc.A (x0, (x1, (x2, (x3, (x4, (x5, ()))))))) := by
  subst h
  funext a j
  fin_cases a <;> fin_cases j <;> rfl

/-- The keys of a member total: the sum of the members' keys. -/
noncomputable def memberKeyF {q m M : Nat} (K : Fin M → Fin 2 → Fin 2 → ExactMatrix q n 1 m) :
    Fin 2 → Fin 2 → ExactMatrix q n 1 m :=
  ∑ j, K j

/-- **The keys of a member total** (`sum.members` at the root): the sum of the members' keys. -/
theorem member_total_keys {q m M : Nat} (hM : 2 ≤ M)
    {fams : Tup (Fin M → ExactMatrix q n 1 m) 6} {a b init final : Tup (ExactMatrix q n 1 m) 6}
    {body : Nat → Tup (ExactMatrix q n 1 m) 6 → Tup (ExactMatrix q n 1 m) 6 → Prop}
    (ha : ∀ c, familyGetStatic (fams.get c) 0 (a.get c))
    (hb : ∀ c, familyGetStatic (fams.get c) 1 (b.get c))
    (hinit : ∀ c, init.get c = a.get c + b.get c)
    (h : MxxIR.IterRuns body (M - 2) init final)
    (hstep : ∀ i cur next, body i cur next → MemberStep.Spec fams (2 + (i : Int)) cur next) :
    Enc.A final = memberKeyF fun j ↦ Enc.A (Tup.ev fams j) :=
  enc_sum_A _ final fun c ↦ by rw [member_total hM ha hb hinit h hstep c]; simp

/-- The keys of column `cc` of a `Step::Sum` row of `W`-member columns. -/
noncomputable def columnKeyF {q m C F W : Nat} (hF : W * C ≤ F)
    (K : Fin F → Fin 2 → Fin 2 → ExactMatrix q n 1 m) (cc : Fin C) :
    Fin 2 → Fin 2 → ExactMatrix q n 1 m :=
  ∑ j : Fin W, K ⟨W * cc + j, by have := cc.isLt; have := j.isLt; nlinarith⟩

/-- **The keys of a column total** (`sum.columns` at the root): block `cc` of the keys is the sum
of the keys of the `W` members of column `cc`. -/
theorem column_total_keys {q m C F W : Nat} (hC : 0 < C) (hF : W * C ≤ F)
    {fams : Tup (Fin F → ExactMatrix q n 1 m) 6}
    {init final : Tup (ExactMatrix q n (1 * 1) (C * m)) 6} {x0 : Tup (ExactMatrix q n 1 m) 6}
    (hx0 : ∀ c, x0.get c = ∑ j ∈ Finset.range W, famAt (fams.get c) (W * 0 + j))
    (hinit : ∀ c, tensorRuns (unitRow ((0 : Nat) : Int) : ExactMatrix q n 1 C) (x0.get c) (init.get c))
    {body : Nat → Tup (ExactMatrix q n (1 * 1) (C * m)) 6 → Tup (ExactMatrix q n (1 * 1) (C * m)) 6 → Prop}
    (h : MxxIR.IterRuns body (C - 1) init final)
    (hstep : ∀ i cur next, body i cur next → ColStep.Spec fams W i cur next) (cc : Fin C) :
    Enc.A (blkT final cc) = columnKeyF hF (fun f ↦ Enc.A (Tup.ev fams f)) cc := by
  have hcol := column_total hC hx0 hinit h hstep
  have hg : ∀ c, (blkT final cc).get c =
      ∑ j : Fin W, (Tup.ev fams ⟨W * cc + j, by have := cc.isLt; have := j.isLt; nlinarith⟩).get c := by
    intro c
    rw [blkT_get, hcol c cc, Finset.sum_range]
    refine Finset.sum_congr rfl fun j _ ↦ ?_
    rw [Tup.get_ev, famAt_of_lt _ (by have := cc.isLt; have := j.isLt; nlinarith)]
  exact enc_sum_A _ _ hg


/-! ## The stand-alone LUT loop step -/

set_option maxHeartbeats 8000000 in
/-- **The keys of one stand-alone LUT loop step** (variant 1: the automorphism `s` is a constant
of the bundle, `sigmas[i]` at the instantiation): the current keys gain `branchKeyF` with
automorphism `s` (input keys `A`, helper keys tagged `s`, carried switching ciphertexts), whatever
the rows; the carried switching ciphertexts become member `i + 1` of their families. -/
theorem LutRoundStep.keys_1 (hn : 0 < n) {q m m' bound W : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {hashModel : HashModel} {base digits : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    {lo0 hi0 lo1 hi1 l2 h2 l3 h3 l4 h4 l5 h5 : Int} {pre : Blob} {s : Int}
    {seed : ByteArray} {i : Nat}
    {r0 r1 k00 k01 k10 k11 b00 b01 b10 b11 : ExactMatrix q n 1 m'}
    {hf0 hf1 : Fin W → ExactMatrix q n 1 m} {x0 a00 a10 x1 a01 a11 : ExactMatrix q n 1 m}
    {sw00 sw01 sw10 sw11 : Fin W → ExactMatrix q n 1 m'}
    {outputs : ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × Unit}
    (h : Dio2Proof.Comp.LutRoundStep.Comp_1 q n m' m' m' m' m' m' m' m' m' m' q q W m W m m m W
      q n m' W q n m' W q n m' W q n m' lo0 hi0 pre s pre s lo1 hi1 pre s pre
      s s s s s s s base digits bound m' m' base digits bound
      m' m' base digits bound m' m' base digits bound m' m' m' m' m' m' l2 h2 l3 h3 l4 h4 l5 h5
      backend hashModel i
      (r0, r1, k00, k01, k10, k11, b00, b01, b10, b11, hf0, seed, hf1, x0, a00, a10, x1, a01, a11,
        sw00, sw01, sw10, sw11, ()) outputs) :
    ![![outputs.2.2.1, outputs.2.2.2.1], ![outputs.2.2.2.2.1, outputs.2.2.2.2.2.1]] =
        ![![k00, k01], ![k10, k11]] + branchKeyF backend base digits
          s.toNat ![![a00, a01], ![a10, a11]]
          (keyPairG hashModel seed q m pre [.integer s])
          ![![b00, b01], ![b10, b11]] ∧
      ∀ hi1 : i + 1 < W, outputs.2.2.2.2.2.2.1 = sw00 ⟨i + 1, hi1⟩ ∧
        outputs.2.2.2.2.2.2.2.1 = sw01 ⟨i + 1, hi1⟩ ∧
        outputs.2.2.2.2.2.2.2.2.1 = sw10 ⟨i + 1, hi1⟩ ∧
        outputs.2.2.2.2.2.2.2.2.2.1 = sw11 ⟨i + 1, hi1⟩ := by
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
    v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49,
    c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18, c19, c20,
    c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37, c38, c39,
    c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, c52, c53, c54, c55, c56, c57, c58,
    c59, c60, c61, c62, h⟩ := h
  subst h
  have hkeys : ![![v1, v6], ![v2, v7]] = keyPairG hashModel seed q m pre
      [.integer s] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact keyPairG_of_hash (pre := [.integer s]) 0 0
        (by norm_num) c4
    · exact keyPairG_of_hash (pre := [.integer s]) 0 1
        (by norm_num) c11
    · exact keyPairG_of_hash (pre := [.integer s]) 1 0
        (by norm_num) c5
    · exact keyPairG_of_hash (pre := [.integer s]) 1 1
        (by norm_num) c12
  have hb := branchKeyF_of_stacks hn hq1 layout hlayout hexact
    (c := ![x0, x1]) (A := ![![a00, a01], ![a10, a11]])
    (rc0 := v11) (rA00 := v12) (rA10 := v13) (rc1 := v16) (rA01 := v17) (rA11 := v18)
    c16 c17 c18 c21 c22 c23 c19 c20 c24 c25 c26
    (sw := ![![b00, b01], ![b10, b11]]) (Xb := ![v22, v32])
    (fun j ↦ by fin_cases j; exacts [c27, c37])
    (D' := ![v23, v33]) (fun j ↦ by fin_cases j; exacts [c28, c38])
    (D := ![v24, v34]) (fun j ↦ by fin_cases j; exacts [c29, c39])
    (M := ![![v25, v35], ![v26, v36]]) (fun j ↦ by fin_cases j; exacts [c30, c40])
    (fun j ↦ by fin_cases j; exacts [c31, c41])
    (Z := ![v27, v37]) (fun j ↦ by fin_cases j; exacts [c32, c42])
    (E' := ![v28, v38]) (fun j ↦ by fin_cases j; exacts [c33, c43])
    (E := ![v29, v39]) (fun j ↦ by fin_cases j; exacts [c34, c44])
    (h := ![v0, v5]) (Ad := ![![v1, v6], ![v2, v7]]) c6 c7 c13 c14 c15
    (K := ![![v42, v43], ![v44, v45]])
    (fun j ↦ by fin_cases j; exacts [c47, c48])
    (fun j ↦ by fin_cases j; exacts [c49, c50])
  refine ⟨?_, fun hi1 ↦ ⟨familyGetDynamic_succ c53 hi1, familyGetDynamic_succ c56 hi1,
    familyGetDynamic_succ c59 hi1, familyGetDynamic_succ c62 hi1⟩⟩
  rw [← hkeys, ← hb]
  funext k j
  fin_cases k <;> fin_cases j <;> rfl


/-! ## The LUT member loop -/

/-- The keys of a dio2 LUT member: the sum over the `W` branches `i` of `branchKeyF` with
automorphism `σ_i = sigmas[i]`, helper keys the root keys of the helper tagged `σ_i` and the
switching ciphertexts `fam · · i`. -/
noncomputable def lutMemberKeyF {q m m' W : Nat} (backend : BackendContext) (base digits : Int)
    (hashModel : HashModel) (seed : ByteArray) (pre : Blob) (sigmas : List Int)
    (A : Fin 2 → Fin 2 → ExactMatrix q n 1 m) (fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q n 1 m') :
    Fin 2 → Fin 2 → ExactMatrix q n 1 m' :=
  ∑ i : Fin W, branchKeyF backend base digits (sigmaOf sigmas i).toNat A
    (keyPairG hashModel seed q m pre [.integer (sigmaOf sigmas i)]) (fun k j ↦ fam k j i)

/-- The closed form of a LUT member's keys from the loop's first `L` branches and the last. -/
theorem lutMemberKeyF_split {q m m' W L : Nat} (hL : L + 1 = W) (backend : BackendContext)
    (base digits : Int)
    (hashModel : HashModel) (seed : ByteArray) (pre : Blob) (sigmas : List Int)
    (A : Fin 2 → Fin 2 → ExactMatrix q n 1 m) (fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q n 1 m')
    (famN : Nat → Fin 2 → Fin 2 → ExactMatrix q n 1 m')
    (hfamN : ∀ i (hi : i < W), famN i = fun k j ↦ fam k j ⟨i, hi⟩) :
    (∑ i ∈ Finset.range L, branchKeyF backend base digits
        (MxxIR.selectInt (Int.ofNat i) sigmas).toNat A
        (keyPairG hashModel seed q m pre [.integer (MxxIR.selectInt (Int.ofNat i) sigmas)]) (famN i)) +
      branchKeyF backend base digits (MxxIR.selectInt (Int.ofNat L) sigmas).toNat A
        (keyPairG hashModel seed q m pre [.integer (MxxIR.selectInt (Int.ofNat L) sigmas)]) (famN L) =
      lutMemberKeyF backend base digits hashModel seed pre sigmas A fam := by
  subst hL
  rw [lutMemberKeyF, Fin.sum_univ_castSucc, Finset.sum_range]
  have hc : ∀ i : Fin L, sigmaOf sigmas (Fin.castSucc i) = MxxIR.selectInt (Int.ofNat i) sigmas :=
    fun i ↦ rfl
  have hl : sigmaOf sigmas (Fin.last L) = MxxIR.selectInt (Int.ofNat L) sigmas := rfl
  simp only [hc, hl]
  refine congrArg₂ (· + ·) (Finset.sum_congr rfl fun i _ ↦ ?_) ?_
  · rw [hfamN i (by omega)]
    exact congrArg _ (funext fun k ↦ funext fun j ↦ congrArg (fam k j) (Fin.ext rfl))
  · rw [hfamN L (by omega)]
    exact congrArg _ (funext fun k ↦ funext fun j ↦ congrArg (fam k j) (Fin.ext rfl))

set_option maxHeartbeats 64000000 in
/-- **The keys of a LUT member at one modulus** (`LutMemberFlat.Comp`): the generated loop over
the first `L = W - 1` branches (each step `LutRoundStep.keys`), the last branch and the finish
compute `lutMemberKeyF` of the input keys `A` and the switching families — whatever the rows,
the helper rows and the LUT's coefficients. -/
theorem LutMemberFlat.keys (hn : 0 < n) {q m m' bound W L : Nat} (hL : L + 1 = W) (hq1 : 1 < q)
    {backend : BackendContext} {hashModel : HashModel} {base digits : Int}
    (layout : RegularLayout q) (hlayout : backend.regularLayout q n = some layout)
    (hexact : layout.droppedModuli = 0)
    {pre : Blob} {sigmas : List Int} {seed : ByteArray}
    {hf0 hf1 : Fin W → ExactMatrix q n 1 m} {x0 a00 a10 x1 a01 a11 : ExactMatrix q n 1 m}
    {sw00 sw01 sw10 sw11 : Fin W → ExactMatrix q n 1 m'}
    {lo0 hi0 lo1 hi1 l2 h2 l3 h3 l4 h4 l5 h5 : Int}
    {child : Nat → ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      (Fin W → ExactMatrix q n 1 m) × ByteArray × (Fin W → ExactMatrix q n 1 m) ×
      ExactMatrix q n 1 m × ExactMatrix q n 1 m × ExactMatrix q n 1 m × ExactMatrix q n 1 m ×
      ExactMatrix q n 1 m × ExactMatrix q n 1 m × (Fin W → ExactMatrix q n 1 m') ×
      (Fin W → ExactMatrix q n 1 m') × (Fin W → ExactMatrix q n 1 m') ×
      (Fin W → ExactMatrix q n 1 m') × Unit →
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × Unit → Prop}
    (hchild : ∀ i a b, child i a b → Dio2Proof.Comp.LutRoundStep.Comp q n m' m' m' m' m' m' m' m'
      m' m' q q W m W m m m W q n m' W q n m' W q n m' W q n m' lo0 hi0 pre sigmas pre sigmas lo1
      hi1 pre sigmas pre sigmas sigmas sigmas sigmas sigmas sigmas sigmas base digits bound m' m'
      base digits bound m' m' base digits bound m' m' base digits bound m' m' m' m' m' m' l2 h2 l3
      h3 l4 h4 l5 h5 backend hashModel i a b)
    {outputs : ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × Unit}
    (h : Dio2Proof.Comp.LutMemberFlat.Comp q n m' q m' q m' W m W m m m W W W W m' m'
      0 0 0 0 (L : Int) (L : Int) pre (L : Int) sigmas pre (L : Int) sigmas (L : Int) pre (L : Int)
      sigmas pre (L : Int) sigmas (L : Int) sigmas (L : Int) sigmas (L : Int) sigmas (L : Int) sigmas
      (L : Int) sigmas (L : Int) sigmas base digits bound m' m' base digits bound m' base digits
      bound m' m' base digits bound m' m' m' m' m' m' m' backend hashModel child
      (sw00, sw01, sw10, sw11, hf0, seed, hf1, x0, a00, a10, x1, a01, a11, ()) outputs) :
    ![![outputs.1, outputs.2.1], ![outputs.2.2.1, outputs.2.2.2.1]] =
      lutMemberKeyF backend base digits hashModel seed pre sigmas ![![a00, a01], ![a10, a11]]
        ![![sw00, sw01], ![sw10, sw11]] := by
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18,
    v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31, v32, v33, v34, v35, v36, v37,
    v38, v39, v40, v41, v42, v43, v44, v45, v46, v47, v48, v49, v50, v51, v52, v53, v54, v55, v56,
    v57, v58, v59, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14, c15, c16, c17, c18,
    c19, c20, c21, c22, c23, c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35, c36, c37,
    c38, c39, c40, c41, c42, c43, c44, c45, c46, c47, c48, c49, c50, c51, h⟩ := h
  subst h
  have hW : 0 < W := by omega
  rw [Int.toNat_natCast] at c5
  -- The switching families as a natural-indexed sequence (zero past the branches).
  let famN : Nat → Fin 2 → Fin 2 → ExactMatrix q n 1 m' := fun i k j ↦
    if hi : i < W then ![![sw00, sw01], ![sw10, sw11]] k j ⟨i, hi⟩ else 0
  have hfamN : ∀ i (hi : i < W), famN i = ![![sw00 ⟨i, hi⟩, sw01 ⟨i, hi⟩],
      ![sw10 ⟨i, hi⟩, sw11 ⟨i, hi⟩]] := by
    intro i hi
    funext k j
    fin_cases k <;> fin_cases j <;> simp [famN, hi]
  -- The loop over the first `L` branches.
  have hloop := carried_loop_gen (S := ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      Unit)
    (fun s ↦ ![![s.2.2.1, s.2.2.2.1], ![s.2.2.2.2.1, s.2.2.2.2.2.1]])
    (fun s ↦ ![![s.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.1], ![s.2.2.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.2.2.1]])
    famN
    (fun i sw ↦ branchKeyF backend base digits (MxxIR.selectInt (Int.ofNat i) sigmas).toNat
      ![![a00, a01], ![a10, a11]]
      (keyPairG hashModel seed q m pre [.integer (MxxIR.selectInt (Int.ofNat i) sigmas)]) sw)
    (by
      rw [hfamN 0 hW]
      funext k j
      fin_cases k <;> fin_cases j
      · exact familyGetDynamic_nat (i := 0) c1 hW
      · exact familyGetDynamic_nat (i := 0) c2 hW
      · exact familyGetDynamic_nat (i := 0) c3 hW
      · exact familyGetDynamic_nat (i := 0) c4 hW)
    (fun i cur nxt hi hbody ↦ by
      obtain ⟨hk, hsw⟩ := LutRoundStep.keys hn hq1 layout hlayout hexact (hchild _ _ _ hbody)
      refine ⟨hk, ?_⟩
      obtain ⟨e0, e1, e2, e3⟩ := hsw (by omega)
      rw [hfamN (i + 1) (by omega)]
      funext k j
      fin_cases k <;> fin_cases j
      · exact e0
      · exact e1
      · exact e2
      · exact e3)
    c5
  obtain ⟨hk, hsw⟩ := hloop
  dsimp only at hk hsw
  -- The last branch.
  have hkeys : ![![v15, v20], ![v16, v21]] = keyPairG hashModel seed q m pre
      [.integer (MxxIR.selectInt (Int.ofNat L) sigmas)] := by
    funext k j
    fin_cases k <;> fin_cases j
    · exact keyPairG_of_hash (pre := [.integer (MxxIR.selectInt (Int.ofNat L) sigmas)]) 0 0
        (by norm_num) c7
    · exact keyPairG_of_hash (pre := [.integer (MxxIR.selectInt (Int.ofNat L) sigmas)]) 0 1
        (by norm_num) c12
    · exact keyPairG_of_hash (pre := [.integer (MxxIR.selectInt (Int.ofNat L) sigmas)]) 1 0
        (by norm_num) c8
    · exact keyPairG_of_hash (pre := [.integer (MxxIR.selectInt (Int.ofNat L) sigmas)]) 1 1
        (by norm_num) c13
  have hb := branchKeyF_of_stacks hn hq1 layout hlayout hexact
    (c := ![x0, x1]) (A := ![![a00, a01], ![a10, a11]]) (index := MxxIR.selectInt (Int.ofNat L) sigmas)
    (rc0 := v25) (rA00 := v26) (rA10 := v27) (rc1 := v30) (rA01 := v31) (rA11 := v32)
    c17 c18 c19 c22 c23 c24 c20 c21 c25 c26 c27
    (sw := ![![v10, v11], ![v12, v13]]) (Xb := ![v36, v45])
    (fun j ↦ by fin_cases j; exacts [c28, c37])
    (D' := ![v37, v46]) (fun j ↦ by fin_cases j; exacts [c29, c38])
    (D := ![v38, v47]) (fun j ↦ by fin_cases j; exacts [c30, c39])
    (M := ![![v39, v48], ![v40, v49]]) (fun j ↦ by fin_cases j; exacts [c31, c40])
    (fun j ↦ by fin_cases j; exacts [c32, c41])
    (Z := ![v41, v50]) (fun j ↦ by fin_cases j; exacts [c33, c42])
    (E' := ![v42, v51]) (fun j ↦ by fin_cases j; exacts [c34, c43])
    (E := ![v43, v52]) (fun j ↦ by fin_cases j; exacts [c35, c44])
    (h := ![v14, v19]) (Ad := ![![v15, v20], ![v16, v21]]) c9 c10 c14 c15 c16
    (K := ![![v44, v53], ![v54, v55]])
    (fun j ↦ by fin_cases j; exacts [c36, c45])
    (fun j ↦ by fin_cases j; exacts [c46, c47])
  rw [hkeys, hsw] at hb
  have hout : ![![matrixAdd v6 v44, matrixAdd v7 v53], ![matrixAdd v8 v54, matrixAdd v9 v55]] =
      ![![v6, v7], ![v8, v9]] + ![![v44, v53], ![v54, v55]] := by
    funext k j
    fin_cases k <;> fin_cases j <;> rfl
  have h0 : (![![(0 : ExactMatrix q n 1 m'), 0], ![0, 0]] : Fin 2 → Fin 2 → _) = 0 := by
    funext k j; fin_cases k <;> fin_cases j <;> rfl
  refine hout.trans ?_
  rw [hk, hb, h0, zero_add]
  exact lutMemberKeyF_split hL backend base digits hashModel seed pre sigmas _ _ famN
    fun i hi ↦ by funext k j; exact dif_pos hi

end Dio2Proof
