import GenBridge_Lut

/-!
# Bundle bridges of the preprocessing's LUT key roles

The key-side counterparts of `GenBridge_Lut` (`GenKey` forms, exact values, no noise):

* `LutKeyRoundStep.rel`: one iteration of a LUT key loop adds the branch keys `lutKeysG … i`
  (`branchKeyG_of_pre`); `LutKeyRoundStep.hstep`: the step for every iteration of a loop whose
  body bundles to `LutKeyRoundStep.Comp`.
* `LutKeyMemberCarrier.rel` (the summed branch keys), `LutKeyMemberReadout.rel` (moved down by the
  block switch), `LutKeyMemberSampler.rel` (plus the offset keys), `SmudgeKeyMember.rel` (moved down,
  times the rebased digits of the weight `c P^(i mod d)`, placed at `X^(⌊i/d⌋ mod N)`).
* `LutKeyTableIndex.rel`, `LutKeyTableStep.rel`/`.rel_1` (+ `.hstep`/`.hstep_1`),
  `LutKeyTable.rel`, `.rel_1`: the switching ciphertexts of a LUT branch, tops `topPairG` and
  `σ(s) top_j + bot_j = (T u_j) G + e_j` with the branch table `T = keyTableG a b (Σ_j F tv j)`
  summed over the table loop (coefficient terms `keyTermC`, exponent terms `keyTermE`) and
  `u = (s, 1)` (`branch_rhs_raw_gen`).
-/

namespace RefreshTopProof.GenIface

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen BrLut
open scoped BigOperators

namespace BrLutKey

variable {n : Nat}

/-- The input tuple of a LUT key-loop body: the four keys, the hash key, the input keys
`(A00, A10)`, the family `sw00`, the input keys `(A01, A11)` and the families `sw10, sw01, sw11`. -/
abbrev KeyStepIn (n q q' m m' W : Nat) :=
  ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
    ByteArray × ExactMatrix q n 1 m × ExactMatrix q n 1 m × (Fin W → ExactMatrix q' n 1 m') ×
    ExactMatrix q n 1 m × ExactMatrix q n 1 m × (Fin W → ExactMatrix q' n 1 m') ×
    (Fin W → ExactMatrix q' n 1 m') × (Fin W → ExactMatrix q' n 1 m') × Unit

/-- The body input of iteration `i` on the keys `cur`. -/
abbrev keyStepIn {q q' m m' W : Nat} (cur : KeysG n q m') (key : ByteArray)
    (a00 a10 : ExactMatrix q n 1 m) (f00 : Fin W → ExactMatrix q' n 1 m') (a01 a11 : ExactMatrix q n 1 m)
    (f10 f01 f11 : Fin W → ExactMatrix q' n 1 m') : KeyStepIn n q q' m m' W :=
  (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, key, a00, a10, f00, a01, a11, f10, f01, f11, ())

/-- The input keys `A`. -/
def keyA {α : Type} (a00 a10 a01 a11 : α) : Fin 2 → Fin 2 → α := ![![a00, a01], ![a10, a11]]

/-- The automorphisms of a LUT table by a natural index. -/
def sigN (tab : List Int) : Nat → Int := fun i ↦ MxxIR.selectInt (Int.ofNat i) tab

/-- The step hypothesis of a key loop with body `child`: iteration `i` adds `F i`. -/
def KeyStepHyp {q q' m m' W : Nat} (child : Nat → KeyStepIn n q q' m m' W → KeysG n q m' → Prop)
    (key : ByteArray) (a00 a10 : ExactMatrix q n 1 m) (f00 : Fin W → ExactMatrix q' n 1 m')
    (a01 a11 : ExactMatrix q n 1 m) (f10 f01 f11 : Fin W → ExactMatrix q' n 1 m')
    (F : Nat → KeysG n q m') : Prop :=
  ∀ i (cur nxt : KeysG n q m'), child i (keyStepIn cur key a00 a10 f00 a01 a11 f10 f01 f11) nxt →
    nxt = cur + F i

/-- The four keys moved down by the block switch. -/
noncomputable def keyBmsG {q q' m : Nat} (basis : List Nat) (x : KeysG n q m) : KeysG n q' m :=
  (blockModSwitchG basis 1 x.1, blockModSwitchG basis 1 x.2.1, blockModSwitchG basis 1 x.2.2.1,
    blockModSwitchG basis 1 x.2.2.2.1, ())

/-- The four keys times `H` and placed at `X^e`. -/
noncomputable def keyPlaceMulG {q m : Nat} (H : ExactMatrix q n m 1) (e : Int) (x : KeysG n q m) :
    KeysG n q 1 :=
  (multiplyMonomial (matrixMul x.1 H) e, multiplyMonomial (matrixMul x.2.1 H) e,
    multiplyMonomial (matrixMul x.2.2.1 H) e, multiplyMonomial (matrixMul x.2.2.2.1 H) e, ())

/-- A key loop from zero keys whose steps satisfy `KeyStepHyp` sums its steps. -/
theorem key_loop_of {q q' m m' W : Nat} {child : Nat → KeyStepIn n q q' m m' W → KeysG n q m' → Prop}
    {key : ByteArray} {a00 a10 a01 a11 : ExactMatrix q n 1 m}
    {f00 f10 f01 f11 : Fin W → ExactMatrix q' n 1 m'} {F : Nat → KeysG n q m'}
    (hchild : KeyStepHyp child key a00 a10 f00 a01 a11 f10 f01 f11 F) {count : Nat}
    {fin : KeysG n q m'}
    (h : MxxIR.IterRuns (fun i current next ↦
      child i (keyStepIn current key a00 a10 f00 a01 a11 f10 f01 f11) next) count
      ((0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'),
        (0 : ExactMatrix q n 1 m'), ()) fin) :
    fin = ∑ i ∈ Finset.range count, F i := by
  have := iter_add_gen F (fun i cur nxt _ hb ↦ hchild i cur nxt hb) h
  rw [this]
  exact zero_add _

/-- Term `j` of a coefficient table (`lut.key-table.step`, variant 0): `lift(t_j) σ(one X^(j s))`
(zero past the `W` entries). -/
noncomputable def keyTermC {q W : Nat} (σ : Int) (tv : Fin W → Int) (one : ExactMatrix q n 1 1)
    (s : Int) (j : Nat) : ExactMatrix q n 1 1 :=
  if h : j < W then
    matrixMulScalarLeft (liftInteger (tv ⟨j, h⟩))
      (ringAutomorphism σ.toNat (multiplyMonomial one (Int.ofNat j * s)))
  else 0

/-- Term `j` of an exponent table (`lut.key-table.step`, variant 1): `σ(one X^(j s)) X^(t_j o)`
(zero past the `W` entries). -/
noncomputable def keyTermE {q W : Nat} (σ : Int) (o : Int) (tv : Fin W → Int)
    (one : ExactMatrix q n 1 1) (s : Int) (j : Nat) : ExactMatrix q n 1 1 :=
  if h : j < W then
    multiplyMonomial (ringAutomorphism σ.toNat (multiplyMonomial one (Int.ofNat j * s)))
      (tv ⟨j, h⟩ * o)
  else 0

/-- The table of a LUT branch: `a (b S)` for the summed terms `S`. -/
noncomputable def keyTableG {q : Nat} (a b : List Int) (S : ExactMatrix q n 1 1) :
    ExactMatrix q n 1 1 :=
  matrixMulScalarLeft (matrixPolynomial a) (matrixMulScalarLeft (matrixPolynomial b) S)

/-- A `seq` loop `sum + term j` from zero sums its terms (any additive state). -/
theorem table_sum_gen {α : Type} [AddCommMonoid α] {body : Nat → α → α → Prop} {count : Nat}
    {final : α} (term : Nat → α) (hstep : ∀ j cur nxt, j < count → body j cur nxt → nxt = cur + term j)
    (h : MxxIR.IterRuns body count 0 final) : final = ∑ j ∈ Finset.range count, term j := by
  rw [iter_add_gen term hstep h, zero_add]

/-- The exported smudge weight `c · lift(P^(i mod d))` is `matrixPolynomial [c P^(i mod d)]`. -/
theorem smudge_weight_eqK {q : Nat} (c P : Int) (i d : Nat) :
    (matrixMulScalarLeft (matrixPolynomial [c])
        (liftInteger (P ^ (Int.fmod (Int.ofNat i) (d : Int)).toNat)) : ExactMatrix q n 1 1) =
      matrixPolynomial [c * P ^ (i % d)] := by
  rw [fmod_ofNat]
  have ht : ((i : Int) % (d : Int)).toNat = i % d := by omega
  funext r k
  simp [matrixMulScalarLeft, matrixPolynomial, liftInteger, ht]

/-- `sigma`, the tag component of a branch table. -/
abbrev sigmaTagB : Blob := [115, 105, 103, 109, 97]
/-- `branch`, the tag component of a branch table. -/
abbrev branchTagB : Blob := [98, 114, 97, 110, 99, 104]

/-- The tag components of the tops of branch `i` with automorphism `σ`. -/
abbrev tableTags (σ : Int) (i : Nat) : List HashTagComponent :=
  [.bytes sigmaTagB, .u64Le σ, .bytes branchTagB, .u64Le (Int.ofNat i)]

/-- The secret and the unit reduced to `q` and the secret's automorphism. -/
theorem table_units {q q1 q2 : Nat} (hq : 1 < q) (hq1 : 1 < q1) (hq2 : 1 < q2) (hn : 0 < n)
    {sp : ErrorPoly n} {sM : ExactMatrix q1 n 1 1} {oneM : ExactMatrix q2 n 1 1}
    (hs : sM = fun _ _ ↦ reducePoly q1 n sp) (hone : oneM = fun _ _ ↦ reducePoly q2 n 1)
    {u0 sσ : ExactMatrix q n 1 1} {u1 : ExactMatrix q n 1 1} {σ : Int}
    (h0 : modulusReduceRuns sM u0) (hσ : ringAutomorphismRuns σ u0 sσ)
    (h1 : modulusReduceRuns oneM u1) :
    u0 = (fun _ _ ↦ reducePoly q n sp) ∧ u1 = (fun _ _ ↦ reducePoly q n 1) ∧
      ∃ hodd : Odd σ.toNat, sσ = fun _ _ ↦ reducePoly q n (substitution σ.toNat hodd sp) := by
  have e0 := modulusReduceRuns_reduce hq1 hq hn h0 (fun _ _ ↦ sp) hs
  have e1 := modulusReduceRuns_reduce hq2 hq hn h1 (fun _ _ ↦ 1) hone
  refine ⟨e0, e1, ?_⟩
  rw [e0] at hσ
  exact ringAutomorphism_reduceG hq hn hσ

end BrLutKey

open BrLutKey

/-! ## One iteration of a LUT key loop -/

set_option maxHeartbeats 64000000 in
/-- **One iteration of a LUT key loop** (`lut.key-round.step`): the bundle on the keys `cur`, the
hash key, the input keys `A = keyA a00 a10 a01 a11` and the switching families adds the branch
keys `lutKeysG … i` of the preprocessing (`branchKeyG_of_pre`). -/
theorem LutKeyRoundStep.rel {n q q' m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {tab : List Int} {key : ByteArray} {level : Int}
    {tag : Blob} {a00 a10 a01 a11 : ExactMatrix q n 1 m}
    {f00 f10 f01 f11 : Fin W → ExactMatrix q' n 1 m'} {i : Nat} {cur nxt : KeysG n q m'}
    {bound : Nat} {basis : List Nat} {mw : Int} {p8 p9 p15 p16 p37 p38 p42 p43 : Int}
    (h : LutKeyRoundStep.Comp q n m' q m' m' m' m' m' m' m' m W q' m W q' W q' W q' m q' m q' q' q'
      level tag tab level tag tab tab tab p8 p9 base digits bound tab tab p15 p16 base digits bound
      mw basis base digits bound level tag tab level tag tab mw basis base digits bound mw
      p37 p38 base digits bound p42 p43 base digits bound mw basis base digits bound
      mw basis base digits bound mw mw mw
      backend hashModel i
      (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, key, a00, a10, f00, a01, a11, f10, f01, f11, ()) nxt)
    (hmw : mw = (m' : Int)) :
    nxt = cur + lutKeysG backend base digits basis hashModel key tag level (sigN tab)
      (keyA a00 a10 a01 a11) (fam4 f00 f01 f10 f11) i := by
  subst hmw
  refine Exists.elim h (fun v0 h ↦ Exists.elim h (fun v1 h ↦ Exists.elim h (fun v2 h ↦ Exists.elim h (fun v3 h ↦ Exists.elim h (fun v4 h ↦ Exists.elim h (fun v5 h ↦ Exists.elim h (fun v6 h ↦ Exists.elim h (fun v7 h ↦ Exists.elim h (fun v8 h ↦ Exists.elim h (fun v9 h ↦ Exists.elim h (fun v10 h ↦ Exists.elim h (fun v11 h ↦ Exists.elim h (fun v12 h ↦ Exists.elim h (fun v13 h ↦ Exists.elim h (fun v14 h ↦ Exists.elim h (fun v15 h ↦ Exists.elim h (fun v16 h ↦ Exists.elim h (fun v17 h ↦ Exists.elim h (fun v18 h ↦ Exists.elim h (fun v19 h ↦ Exists.elim h (fun v20 h ↦ Exists.elim h (fun v21 h ↦ Exists.elim h (fun v22 h ↦ Exists.elim h (fun v23 h ↦ Exists.elim h (fun v24 h ↦ Exists.elim h (fun v25 h ↦ Exists.elim h (fun v26 h ↦ Exists.elim h (fun v27 h ↦ Exists.elim h (fun v28 h ↦ Exists.elim h (fun v29 h ↦ Exists.elim h (fun v30 h ↦ Exists.elim h (fun v31 h ↦ Exists.elim h (fun v32 h ↦ Exists.elim h (fun v33 h ↦ Exists.elim h (fun v34 h ↦ Exists.elim h (fun v35 h ↦ Exists.elim h (fun v36 h ↦ Exists.elim h (fun v37 h ↦ Exists.elim h (fun v38 h ↦ Exists.elim h (fun v39 h ↦ Exists.elim h (fun v40 h ↦ Exists.elim h (fun v41 h ↦ Exists.elim h (fun v42 h ↦ Exists.elim h (fun v43 h ↦ ?_))))))))))))))))))))))))))))))))))))))))))))
  obtain ⟨t0, t1, t2, t3, t4, t5, -, -, t8, t9, t10, t11, t12, t13, -, -, t16, t17, t18, t19,
    t20, t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, -, -, t33, t34, t35, -, -, t38, t39,
    t40, t41, t42, t43, t44, t45, t46, t47, t48, t49, t50, t51, rfl⟩ := h
  obtain ⟨hi, rfl⟩ := familyGet_lit i t8
  obtain ⟨_, rfl⟩ := familyGet_lit i t16
  obtain ⟨_, rfl⟩ := familyGet_lit i t33
  obtain ⟨_, rfl⟩ := familyGet_lit i t38
  obtain ⟨rfl, rfl, rfl, rfl⟩ := lutA_of_samples t0 t23 t1 t24
  have hK := branchKeyG_of_pre (A := keyA a00 a10 a01 a11) (index := sigN tab i)
    (Ad := lutAG hashModel key q m tag level (sigN tab i))
    (sw := fun k j ↦ fam4 f00 f01 f10 f11 k j ⟨i, hi⟩) t3 t4 t11 t12 t5 t13 t2 t25
    (dA := ![v7, v28]) (Fin.forall_fin_two.2 ⟨t9, t34⟩)
    (rA := ![v8, v29]) (Fin.forall_fin_two.2 ⟨t10, t35⟩)
    (dB := ![v13, v31]) (Fin.forall_fin_two.2 ⟨t17, t39⟩)
    (rB := ![v14, v32]) (Fin.forall_fin_two.2 ⟨t18, t40⟩)
    (M := ![![v15, v33], ![v22, v37]]) (Fin.forall_fin_two.2 ⟨t19, t41⟩)
    (Fin.forall_fin_two.2 ⟨t26, t45⟩) (Asw := ![![v16, v34], ![v23, v38]])
    (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨t20, t42⟩, Fin.forall_fin_two.2 ⟨t27, t46⟩⟩)
    (dC := ![v17, v35]) (Fin.forall_fin_two.2 ⟨t21, t43⟩)
    (rC := ![v18, v36]) (Fin.forall_fin_two.2 ⟨t22, t44⟩)
    (dE := ![v24, v39]) (Fin.forall_fin_two.2 ⟨t28, t47⟩)
    (rE := ![v25, v40]) (Fin.forall_fin_two.2 ⟨t29, t48⟩)
    (K := ![![v26, v41], ![v42, v43]]) (Fin.forall_fin_two.2 ⟨t30, t49⟩)
    (Fin.forall_fin_two.2 ⟨t50, t51⟩)
  rw [lutKeysG, dif_pos hi, ← hK]
  rfl

/-- **The step of every iteration** of a key loop whose body `child` bundles to
`LutKeyRoundStep.Comp` (`hcomp`: the child scope's adapter). -/
theorem LutKeyRoundStep.hstep {n q q' m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {tab : List Int} {key : ByteArray} {level : Int}
    {tag : Blob} {a00 a10 a01 a11 : ExactMatrix q n 1 m}
    {f00 f10 f01 f11 : Fin W → ExactMatrix q' n 1 m'}
    {child : Nat → KeyStepIn n q q' m m' W → KeysG n q m' → Prop}
    {bound : Nat} {basis : List Nat} {mw : Int} {p8 p9 p15 p16 p37 p38 p42 p43 : Int}
    (hcomp : ∀ i (cur nxt : KeysG n q m'),
      child i (keyStepIn cur key a00 a10 f00 a01 a11 f10 f01 f11) nxt →
      LutKeyRoundStep.Comp q n m' q m' m' m' m' m' m' m' m W q' m W q' W q' W q' m q' m q' q' q'
        level tag tab level tag tab tab tab p8 p9 base digits bound tab tab p15 p16 base digits bound
        mw basis base digits bound level tag tab level tag tab mw basis base digits bound mw
        p37 p38 base digits bound p42 p43 base digits bound mw basis base digits bound
        mw basis base digits bound mw mw mw
        backend hashModel i
        (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, key, a00, a10, f00, a01, a11, f10, f01, f11, ()) nxt)
    (hmw : mw = (m' : Int)) :
    KeyStepHyp child key a00 a10 f00 a01 a11 f10 f01 f11
      (lutKeysG backend base digits basis hashModel key tag level (sigN tab)
        (keyA a00 a10 a01 a11) (fam4 f00 f01 f10 f11)) :=
  fun i cur nxt hc ↦ LutKeyRoundStep.rel (hcomp i cur nxt hc) hmw

/-! ## LUT key members -/

/-- **The summed branch keys of a LUT** (`lut.key-member.carrier`): `Σ_{i < count} lutKeysG … i`. -/
theorem LutKeyMemberCarrier.rel {n q q' m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {key : ByteArray} {a00 a10 a01 a11 : ExactMatrix q n 1 m}
    {f00 f10 f01 f11 : Fin W → ExactMatrix q' n 1 m'}
    {child : Nat → KeyStepIn n q q' m m' W → KeysG n q m' → Prop} {i : Nat} {cnt : Int}
    {outputs : KeysG n q m'}
    (h : LutKeyMemberCarrier.Comp q n m' q n m q n m W q' n m' q n m q n m W q' n m' W q' n m'
      W q' n m' cnt backend hashModel i child (key, a00, a10, f00, a01, a11, f10, f01, f11, ())
      outputs)
    {F : Nat → KeysG n q m'} (hchild : KeyStepHyp child key a00 a10 f00 a01 a11 f10 f01 f11 F) :
    outputs = ∑ k ∈ Finset.range cnt.toNat, F k := by
  obtain ⟨v0, v1, v2, v3, a0, rfl⟩ := h
  exact key_loop_of hchild a0

/-- **The readout keys of a LUT member** (`lut.key-member.readout`): the summed branch keys moved
down by the block switch. -/
theorem LutKeyMemberReadout.rel {n q q' m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {key : ByteArray} {a00 a10 a01 a11 : ExactMatrix q n 1 m}
    {f00 f10 f01 f11 : Fin W → ExactMatrix q' n 1 m'}
    {child : Nat → KeyStepIn n q q' m m' W → KeysG n q m' → Prop} {i : Nat} {cnt : Int}
    {basis' : List Nat} {outputs : KeysG n q' m'}
    (h : LutKeyMemberReadout.Comp q n m' q n m q n m W q' n m' q n m q n m W q' n m' W q' n m'
      W q' n m' q' q' q' q' cnt basis' basis' basis' basis' backend hashModel i child
      (key, a00, a10, f00, a01, a11, f10, f01, f11, ()) outputs)
    {F : Nat → KeysG n q m'} (hchild : KeyStepHyp child key a00 a10 f00 a01 a11 f10 f01 f11 F) :
    outputs = keyBmsG basis' (∑ k ∈ Finset.range cnt.toNat, F k) := by
  obtain ⟨v0, v1, v2, v3, v4, v5, v6, v7, a0, a1, a2, a3, a4, rfl⟩ := h
  have hf := key_loop_of hchild a0
  rw [← hf, blockModSwitchRuns_eqG a1, blockModSwitchRuns_eqG a2, blockModSwitchRuns_eqG a3,
    blockModSwitchRuns_eqG a4]
  rfl

/-- **The sampler keys of a LUT member** (`lut.key-member.sampler`): the summed branch keys plus
the offset keys `(o0, o1, o2, o3)`. -/
theorem LutKeyMemberSampler.rel {n q q' m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {key : ByteArray} {a00 a10 a01 a11 : ExactMatrix q n 1 m}
    {f00 f10 f01 f11 : Fin W → ExactMatrix q' n 1 m'} {o0 o1 o2 o3 : ExactMatrix q n 1 m'}
    {child : Nat → KeyStepIn n q q' m m' W → KeysG n q m' → Prop} {i : Nat} {cnt : Int}
    {outputs : KeysG n q m'}
    (h : LutKeyMemberSampler.Comp q n m' q n m q n m W q' n m' q n m q n m W q' n m' W q' n m'
      W q' n m' cnt backend hashModel i child
      (key, a00, a10, f00, a01, a11, f10, f01, f11, o0, o1, o2, o3, ()) outputs)
    {F : Nat → KeysG n q m'} (hchild : KeyStepHyp child key a00 a10 f00 a01 a11 f10 f01 f11 F) :
    outputs = (∑ k ∈ Finset.range cnt.toNat, F k) + (o0, o1, o2, o3, ()) := by
  obtain ⟨v0, v1, v2, v3, a0, rfl⟩ := h
  rw [← key_loop_of hchild a0]
  rfl

set_option maxHeartbeats 64000000 in
/-- **The keys of a smudge member** (`smudge.key-member`): the summed branch keys moved down,
times the rebased digits `rdecG` of the weight `c P^(i mod d)`, placed at `X^(⌊i/d⌋ mod N)` (the
keys of `SmudgeMember.rel`'s output). -/
theorem SmudgeKeyMember.rel {n q q' m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {key : ByteArray} {a00 a10 a01 a11 : ExactMatrix q n 1 m}
    {f00 f10 f01 f11 : Fin W → ExactMatrix q' n 1 m'}
    {child : Nat → KeyStepIn n q q' m m' W → KeysG n q m' → Prop} {i : Nat} {cnt : Int}
    {basis' : List Nat} {baseG digitsG : Int} {boundG : Nat} {D NN c P : Int}
    {outputs : KeysG n q' 1}
    (h : SmudgeKeyMember.Comp q n m' q n m q n m W q' n m' q n m q n m W q' n m' W q' n m'
      W q' n m' q' q' cnt basis' baseG digitsG [c] P D boundG basis' basis' basis'
      D NN D NN D NN D NN backend hashModel i child
      (key, a00, a10, f00, a01, a11, f10, f01, f11, ()) outputs)
    (d : Nat) (hD : D = (d : Int)) (N : Nat) (hNN : NN = (N : Int))
    {F : Nat → KeysG n q m'} (hchild : KeyStepHyp child key a00 a10 f00 a01 a11 f10 f01 f11 F) :
    outputs = keyPlaceMulG (rdecG (inner := m') backend baseG digitsG
        (matrixPolynomial [c * P ^ (i % d)] : ExactMatrix q' n 1 1))
        ((i / d % N : Nat) : Int) (keyBmsG basis' (∑ k ∈ Finset.range cnt.toNat, F k)) := by
  subst hD hNN
  refine Exists.elim h (fun v0 h ↦ Exists.elim h (fun v1 h ↦ Exists.elim h (fun v2 h ↦ Exists.elim h (fun v3 h ↦ Exists.elim h (fun v4 h ↦ Exists.elim h (fun v5 h ↦ Exists.elim h (fun v6 h ↦ Exists.elim h (fun v7 h ↦ Exists.elim h (fun v8 h ↦ Exists.elim h (fun v9 h ↦ ?_))))))))))
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, hout⟩ := h
  rw [smudge_weight_eqK] at a2
  rw [fdiv_fmod_ofNat] at hout
  subst hout
  have hf := key_loop_of hchild a0
  rw [← hf, ← rdecG_eq a2 a3, blockModSwitchRuns_eqG a1, blockModSwitchRuns_eqG a4,
    blockModSwitchRuns_eqG a5, blockModSwitchRuns_eqG a6]
  rfl

/-! ## The switching ciphertexts of a LUT branch -/

/-- **A table index** (`lut.key-table.index`): entry `j` is `selectInt j vals + off`. -/
theorem LutKeyTableIndex.rel {vals : List Int} {off : Int} {backend : BackendContext}
    {hashModel : HashModel} {j : Nat} {v : Int}
    (h : LutKeyTableIndex.Comp vals off backend hashModel j () v) :
    v = MxxIR.selectInt (Int.ofNat j) vals + off := h

/-- **One term of a coefficient table** (`lut.key-table.step`, variant 0): iteration `j` of the
table loop of branch `j1` adds `keyTermC σ_{j1} tv one s j`. -/
theorem LutKeyTableStep.rel {n q W : Nat} {lo hi : Int} {tab : List Int}
    {backend : BackendContext} {hashModel : HashModel} {j1 j : Nat}
    {cur nxt one : ExactMatrix q n 1 1} {tv : Fin W → Int} {s : Int}
    (h : LutKeyTableStep.Comp q n W lo hi tab backend hashModel j1 j (cur, tv, one, s, ()) nxt) :
    nxt = cur + keyTermC (MxxIR.selectInt (Int.ofNat j1) tab) tv one s j := by
  obtain ⟨v0, v1, -, -, a2, a3, rfl⟩ := h
  obtain ⟨hj, rfl⟩ := familyGet_lit j a2
  obtain rfl := a3.2.2.2
  rw [keyTermC, dif_pos hj]
  rfl

/-- **One term of an exponent table** (`lut.key-table.step`, variant 1): iteration `j` adds
`keyTermE σ_{j1} o tv one s j`. -/
theorem LutKeyTableStep.rel_1 {n q W : Nat} {lo hi o : Int} {tab : List Int}
    {backend : BackendContext} {hashModel : HashModel} {j1 j : Nat}
    {cur nxt one : ExactMatrix q n 1 1} {tv : Fin W → Int} {s : Int}
    (h : LutKeyTableStep.Comp_1 q n W tab lo hi o backend hashModel j1 j (cur, one, s, tv, ()) nxt) :
    nxt = cur + keyTermE (MxxIR.selectInt (Int.ofNat j1) tab) o tv one s j := by
  obtain ⟨v0, v1, a0, -, -, a3, rfl⟩ := h
  obtain ⟨hj, rfl⟩ := familyGet_lit j a3
  obtain rfl := a0.2.2.2
  rw [keyTermE, dif_pos hj]
  rfl

/-- The steps of a coefficient-table loop whose body bundles to `LutKeyTableStep.Comp`
(`hcomp`: the step adapter), for every table family `tv`. -/
theorem LutKeyTableStep.hstep {n q W : Nat} {lo hi : Int} {tab : List Int}
    {backend : BackendContext} {hashModel : HashModel} {j1 : Nat} {one : ExactMatrix q n 1 1}
    {s : Int}
    {child : Nat → ExactMatrix q n 1 1 × (Fin W → Int) × ExactMatrix q n 1 1 × Int × Unit →
      ExactMatrix q n 1 1 → Prop}
    (hcomp : ∀ (tv : Fin W → Int) j (cur nxt : ExactMatrix q n 1 1), child j (cur, tv, one, s, ()) nxt →
      LutKeyTableStep.Comp q n W lo hi tab backend hashModel j1 j (cur, tv, one, s, ()) nxt) :
    ∀ (tv : Fin W → Int) j cur nxt, child j (cur, tv, one, s, ()) nxt →
      nxt = cur + keyTermC (MxxIR.selectInt (Int.ofNat j1) tab) tv one s j :=
  fun tv j cur nxt hc ↦ LutKeyTableStep.rel (hcomp tv j cur nxt hc)

/-- The steps of an exponent-table loop whose body bundles to `LutKeyTableStep.Comp_1`. -/
theorem LutKeyTableStep.hstep_1 {n q W : Nat} {lo hi o : Int} {tab : List Int}
    {backend : BackendContext} {hashModel : HashModel} {j1 : Nat} {one : ExactMatrix q n 1 1}
    {s : Int}
    {child : Nat → ExactMatrix q n 1 1 × ExactMatrix q n 1 1 × Int × (Fin W → Int) × Unit →
      ExactMatrix q n 1 1 → Prop}
    (hcomp : ∀ (tv : Fin W → Int) j (cur nxt : ExactMatrix q n 1 1), child j (cur, one, s, tv, ()) nxt →
      LutKeyTableStep.Comp_1 q n W tab lo hi o backend hashModel j1 j (cur, one, s, tv, ()) nxt) :
    ∀ (tv : Fin W → Int) j cur nxt, child j (cur, one, s, tv, ()) nxt →
      nxt = cur + keyTermE (MxxIR.selectInt (Int.ofNat j1) tab) o tv one s j :=
  fun tv j cur nxt hc ↦ LutKeyTableStep.rel_1 (hcomp tv j cur nxt hc)

set_option maxHeartbeats 64000000 in
/-- **The switching ciphertexts of LUT branch `i`** (`lut.key-table`, variant 0: a coefficient
table, the gadget at the working modulus): the tops are hash samples and
`σ(s) top_j + bot_j = (T u_j) G + e_j` with `T = keyTableG a b (Σ_{j < cnt} F tv j)`, the table
family `tv` (`htable`, from `LutKeyTableIndex.rel`), the loop's terms `F` (`hchild`, from
`LutKeyTableStep.hstep`), `u = (s, 1)` and `e_j` within the Gaussian cutoff. -/
theorem LutKeyTable.rel {n q q2 q3 m' W : Nat} {backend : BackendContext} {hashModel : HashModel}
    {base digits : Int} {tab : List Int} {tag : Blob} {oneL : List Int} {stride cnt : Int}
    {a b : List Int} {sigma : Rat} {cutoff : Int} {i : Nat} {key : ByteArray}
    {child : Nat → ExactMatrix q n 1 1 × (Fin W → Int) × ExactMatrix q n 1 1 × Int × Unit →
      ExactMatrix q n 1 1 → Prop} {table : Nat → Int → Prop}
    {sM : ExactMatrix q2 n 1 1} {oneM : ExactMatrix q3 n 1 1}
    {top0 top1 bot0 bot1 : ExactMatrix q n 1 m'}
    (h : LutKeyTable.Comp W q n q n m' q2 q3 tab tag tab tag tab oneL stride cnt base digits
      sigma cutoff sigma cutoff a b a b backend hashModel i child table
      (key, sM, oneM, ()) (top0, top1, bot0, bot1, ()))
    (tv : Fin W → Int) (htable : ∀ (j : Fin W) v, table j v → v = tv j)
    (F : (Fin W → Int) → Nat → ExactMatrix q n 1 1)
    (hchild : ∀ (tv' : Fin W → Int) j cur nxt,
      child j (cur, tv', matrixPolynomial oneL, -stride, ()) nxt → nxt = cur + F tv' j)
    (hq : 1 < q) (hq2 : 1 < q2) (hq3 : 1 < q3) (hn : 0 < n) {sp : ErrorPoly n}
    (hs : sM = fun _ _ ↦ reducePoly q2 n sp) (hone : oneM = fun _ _ ↦ reducePoly q3 n 1) :
    top0 = topPairG hashModel key q m' tag (tableTags (MxxIR.selectInt (Int.ofNat i) tab) i) 0 ∧
      top1 = topPairG hashModel key q m' tag (tableTags (MxxIR.selectInt (Int.ofNat i) tab) i) 1 ∧
      ∃ G : ExactMatrix q n 1 m', gadgetMatrixRuns backend base digits G ∧
      ∃ hodd : Odd (MxxIR.selectInt (Int.ofNat i) tab).toNat,
      ∃ eB : Fin 2 → ErrorMatrix n 1 m', (∀ j, CoeffBound (eB j) cutoff.toNat) ∧ ∀ j,
        reducePoly q n (substitution (MxxIR.selectInt (Int.ofNat i) tab).toNat hodd sp) •
            ![top0, top1] j + ![bot0, bot1] j =
          ((keyTableG a b (∑ k ∈ Finset.range cnt.toNat, F tv k) : ExactMatrix q n 1 1) 0 0 *
              reducePoly q n (unitPair sp j)) • G + reduceMatrix q n 1 m' (eB j) := by
  refine Exists.elim h (fun v0 h ↦ Exists.elim h (fun v1 h ↦ Exists.elim h (fun v2 h ↦ Exists.elim h (fun v3 h ↦ Exists.elim h (fun v4 h ↦ Exists.elim h (fun v5 h ↦ Exists.elim h (fun v6 h ↦ Exists.elim h (fun v7 h ↦ Exists.elim h (fun v8 h ↦ Exists.elim h (fun v9 h ↦ ?_))))))))))
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, hout⟩ := h
  simp only [Prod.mk.injEq] at hout
  obtain ⟨rfl, rfl, rfl, rfl, -⟩ := hout
  obtain rfl : v4 = tv := funext fun j ↦ htable j _ (a4 j)
  obtain rfl := table_sum_gen (F v4) (fun j cur nxt _ hb ↦ hchild v4 j cur nxt hb) a5
  obtain ⟨e0, e1, hodd, e3⟩ := table_units hq hq2 hq3 hn hs hone a2 a3 a8
  subst e0 e1 e3
  obtain ⟨E0, hE0, h0⟩ := branch_rhs_raw_gen (u := sp) rfl rfl (top := top0) a7
  obtain ⟨E1, hE1, h1⟩ := branch_rhs_raw_gen (u := 1) rfl rfl (top := top1) a9
  refine ⟨topPairG_of_hash (pre := tableTags _ i) 0 (by norm_num) a0,
    topPairG_of_hash (pre := tableTags _ i) 1 (by norm_num) a1, v6, a6, hodd, ![E0, E1],
    Fin.forall_fin_two.2 ⟨hE0, hE1⟩, Fin.forall_fin_two.2 ⟨h0, h1⟩⟩

set_option maxHeartbeats 64000000 in
/-- **The switching ciphertexts of LUT branch `i`** (variant 1: an exponent table, the gadget
rebased from `q4`); as `LutKeyTable.rel` with the loop's input order `(cur, one, -s, tv)`. -/
theorem LutKeyTable.rel_1 {n q q2 q3 q4 m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {tab : List Int} {tag : Blob} {oneL : List Int}
    {stride cnt : Int} {a b : List Int} {sigma : Rat} {cutoff : Int} {i : Nat} {key : ByteArray}
    {child : Nat → ExactMatrix q n 1 1 × ExactMatrix q n 1 1 × Int × (Fin W → Int) × Unit →
      ExactMatrix q n 1 1 → Prop} {table : Nat → Int → Prop}
    {sM : ExactMatrix q2 n 1 1} {oneM : ExactMatrix q3 n 1 1}
    {top0 top1 bot0 bot1 : ExactMatrix q n 1 m'}
    (h : LutKeyTable.Comp_1 W q n q n m' q2 q3 q4 tab tag tab tag tab oneL stride cnt base digits
      sigma cutoff sigma cutoff a b a b backend hashModel i child table
      (key, sM, oneM, ()) (top0, top1, bot0, bot1, ()))
    (tv : Fin W → Int) (htable : ∀ (j : Fin W) v, table j v → v = tv j)
    (F : (Fin W → Int) → Nat → ExactMatrix q n 1 1)
    (hchild : ∀ (tv' : Fin W → Int) j cur nxt,
      child j (cur, matrixPolynomial oneL, -stride, tv', ()) nxt → nxt = cur + F tv' j)
    (hq : 1 < q) (hq2 : 1 < q2) (hq3 : 1 < q3) (hn : 0 < n) {sp : ErrorPoly n}
    (hs : sM = fun _ _ ↦ reducePoly q2 n sp) (hone : oneM = fun _ _ ↦ reducePoly q3 n 1) :
    top0 = topPairG hashModel key q m' tag (tableTags (MxxIR.selectInt (Int.ofNat i) tab) i) 0 ∧
      top1 = topPairG hashModel key q m' tag (tableTags (MxxIR.selectInt (Int.ofNat i) tab) i) 1 ∧
      ∃ (G0 : ExactMatrix q4 n 1 m') (G : ExactMatrix q n 1 m'),
        gadgetMatrixRuns backend base digits G0 ∧ centeredRebaseRuns G0 G ∧
      ∃ hodd : Odd (MxxIR.selectInt (Int.ofNat i) tab).toNat,
      ∃ eB : Fin 2 → ErrorMatrix n 1 m', (∀ j, CoeffBound (eB j) cutoff.toNat) ∧ ∀ j,
        reducePoly q n (substitution (MxxIR.selectInt (Int.ofNat i) tab).toNat hodd sp) •
            ![top0, top1] j + ![bot0, bot1] j =
          ((keyTableG a b (∑ k ∈ Finset.range cnt.toNat, F tv k) : ExactMatrix q n 1 1) 0 0 *
              reducePoly q n (unitPair sp j)) • G + reduceMatrix q n 1 m' (eB j) := by
  refine Exists.elim h (fun v0 h ↦ Exists.elim h (fun v1 h ↦ Exists.elim h (fun v2 h ↦ Exists.elim h (fun v3 h ↦ Exists.elim h (fun v4 h ↦ Exists.elim h (fun v5 h ↦ Exists.elim h (fun v6 h ↦ Exists.elim h (fun v7 h ↦ Exists.elim h (fun v8 h ↦ Exists.elim h (fun v9 h ↦ Exists.elim h (fun v10 h ↦ ?_)))))))))))
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, hout⟩ := h
  simp only [Prod.mk.injEq] at hout
  obtain ⟨rfl, rfl, rfl, rfl, -⟩ := hout
  obtain rfl : v4 = tv := funext fun j ↦ htable j _ (a4 j)
  obtain rfl := table_sum_gen (F v4) (fun j cur nxt _ hb ↦ hchild v4 j cur nxt hb) a5
  obtain ⟨e0, e1, hodd, e3⟩ := table_units hq hq2 hq3 hn hs hone a2 a3 a9
  subst e0 e1 e3
  obtain ⟨E0, hE0, h0⟩ := branch_rhs_raw_gen (u := sp) rfl rfl (top := top0) a8
  obtain ⟨E1, hE1, h1⟩ := branch_rhs_raw_gen (u := 1) rfl rfl (top := top1) a10
  refine ⟨topPairG_of_hash (pre := tableTags _ i) 0 (by norm_num) a0,
    topPairG_of_hash (pre := tableTags _ i) 1 (by norm_num) a1, v6, v7, a6, a7, hodd, ![E0, E1],
    Fin.forall_fin_two.2 ⟨hE0, hE1⟩, Fin.forall_fin_two.2 ⟨h0, h1⟩⟩

end RefreshTopProof.GenIface
