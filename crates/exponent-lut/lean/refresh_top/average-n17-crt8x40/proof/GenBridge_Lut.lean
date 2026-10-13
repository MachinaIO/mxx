import GenIface
import GenRelLut
import GenKey

/-!
# Bundle bridges of the online LUT and smudge roles

Each bridge destructures a computation bundle `X.Comp` of `GenIface` (the scope relation with the
instance constants as parameters) and calls the encoding-level lemma of `GenRelLut`:

* `LutRoundStep.rel`: one iteration of a LUT branch loop (`lut.round.step`) is the step hypothesis
  of `lut_loop_gen` (`lut_step_gen`); `LutRoundStep.keys`: its keys are the preprocessing's
  branch keys `branchKeyG` (`branchKeyG_of_stacks`, the LUT key link); `LutRoundStep.hstep`: the
  step hypothesis for every iteration of a loop whose body bundles to `LutRoundStep.Comp`.
* `LutMemberReadout.rel` (loop, last branch and move down: `lut_round_gen`),
  `LutMemberSampler.rel` (loop and last branch: `lut_branches_gen`), `LutMemberSamplerOffset.rel`
  (the sampler plus the offset encoding), `SmudgeMember.rel` (the readout finalized with the weight
  `c P^(i mod d)` and placed at `X^(⌊i/d⌋ mod N)`: `smudge_member_weight_gen`).

Bundle parameters that the lemmas need to be one value (the automorphism table, the hash tag, the
gadget base and digits, the decomposition bound, the CRT basis, the widths) are instantiated with
one variable each, so the instance's repeated literals unify; the slice widths are an `Int`
variable `mw` with `hmw : mw = m'`. The bounds of `familyGetDynamic` reads stay free.
-/

namespace RefreshTopProof.GenIface

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen
open scoped BigOperators

namespace BrLut

variable {n : Nat}

/-- The automorphisms of a LUT read from its table: branch `k` is `selectInt k tab`. -/
def sig (tab : List Int) (W : Nat) : Fin W → Int := fun k ↦ MxxIR.selectInt (Int.ofNat k) tab

/-- Four switching families as `fam k j`. -/
def fam4 {α : Type} {W : Nat} (f00 f01 f10 f11 : Fin W → α) : Fin 2 → Fin 2 → Fin W → α :=
  ![![f00, f01], ![f10, f11]]

/-- The keys of a LUT input given as `(c0, A00, A10, c1, A01, A11)`. -/
def inA {α : Type} (_x0 x1 x2 _x3 x4 x5 : α) : Fin 2 → Fin 2 → α := ![![x1, x4], ![x2, x5]]

/-- The rows of a LUT input given as `(c0, A00, A10, c1, A01, A11)`. -/
def inC {α : Type} (x0 _x1 _x2 x3 _x4 _x5 : α) : Fin 2 → α := ![x0, x3]

/-- A family read at `i + 1`. -/
theorem familyGet_succ {α : Type} {count : Nat} {f : Fin count → α} (i : Nat) {out : α}
    (h : familyGetDynamic f (Int.ofNat i + 1) out) : ∃ hi : i + 1 < count, out = f ⟨i + 1, hi⟩ := by
  obtain ⟨pos, hpos, hout⟩ := h
  have hv : pos.val = i + 1 := by simp only [Int.ofNat_eq_natCast] at hpos; omega
  exact ⟨hv ▸ pos.isLt, by rw [hout]; exact congrArg f (Fin.ext hv)⟩

/-- A family read at the literal `k`. -/
theorem familyGet_lit {α : Type} {count : Nat} {f : Fin count → α} (k : Nat) {out : α}
    (h : familyGetDynamic f (Int.ofNat k) out) : ∃ hk : k < count, out = f ⟨k, hk⟩ :=
  familyGet_ofNatG k h

/-- The four helper samples of a LUT branch are the helper keys `lutAG`. -/
theorem lutA_of_samples {q m : Nat} {model : HashModel} {key : ByteArray} {tag : Blob}
    {level σ : Int} {a00 a01 a10 a11 : ExactMatrix q n 1 m}
    (h00 : hashSample model tag [.u64Le level, .integer σ, .bytes componentTagG, .u64Le 0] key a00)
    (h01 : hashSample model tag [.u64Le level, .integer σ, .bytes componentTagG, .u64Le 1] key a01)
    (h10 : hashSample model tag [.u64Le level, .integer σ, .bytes componentTagG, .u64Le 2] key a10)
    (h11 : hashSample model tag [.u64Le level, .integer σ, .bytes componentTagG, .u64Le 3] key a11) :
    a00 = lutAG model key q m tag level σ 0 0 ∧ a01 = lutAG model key q m tag level σ 0 1 ∧
      a10 = lutAG model key q m tag level σ 1 0 ∧ a11 = lutAG model key q m tag level σ 1 1 :=
  ⟨keyPairG_of_hash (pre := [.u64Le level, .integer σ]) 0 0 (by norm_num) h00,
    keyPairG_of_hash (pre := [.u64Le level, .integer σ]) 0 1 (by norm_num) h01,
    keyPairG_of_hash (pre := [.u64Le level, .integer σ]) 1 0 (by norm_num) h10,
    keyPairG_of_hash (pre := [.u64Le level, .integer σ]) 1 1 (by norm_num) h11⟩

end BrLut

open BrLut

/-! ## One iteration of a LUT branch loop -/

set_option maxHeartbeats 64000000 in

/-- **One iteration of a LUT branch loop** (`lut.round.step`): the bundle with the state `cur`
(rows, keys, carried switching ciphertexts of branch `i`), the helper families `hf0`, `hf1` under
the hash key `key`, the LUT input `(x0, …, x5)` = `(c0, A00, A10, c1, A01, A11)` and the four
switching families `f00 … f11` adds branch `i` and carries branch `i + 1`: the step hypothesis of
`lut_loop_gen`. The helper of branch `k` encodes `p σ_k(s)` under the hash-sampled keys
`lutAG … level σ_k` (`hhelp`). -/
theorem LutRoundStep.rel {n q q' m m' W : Nat} {backend : BackendContext} {hashModel : HashModel}
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
        x0, x1, x2, x3, x4, x5, f00, f01, f10, f11, ()) nxt)
    (hmw : mw = (m' : Int)) (hn : 0 < n) {log : Nat} (hpow : n = 2 ^ log)
    {p : Nat} (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q')
    (layout : RegularLayout q') (hlayout : backend.regularLayout q' n = some layout)
    (hexact : layout.droppedModuli = 0) (hodd : ∀ k, Odd (sig tab W k).toNat)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix q' n 1 m} {g : ExactMatrix q n 1 m}
    (hg' : gadgetMatrixRuns backend base digits g') (hg : centeredRebaseRuns g' g) {B : Nat}
    (hinput : Rel sm (inA x0 x1 x2 x3 x4 x5)
      (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) (inC x0 x1 x2 x3 x4 x5) B)
    {Bh : Nat}
    (hhelp : ∀ k : Fin W, Rel sm (lutAG hashModel key q m tag level (sig tab W k))
      (fun j ↦ reducePoly q n ((p : Int) *
        unitPair (substitution (sig tab W k).toNat (hodd k) sm) j) • g) ![hf0 k, hf1 k] Bh)
    (hi : i + 1 < W) {Tc : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hcur : Rel sm cur.key Tc cur.row Bs)
    (hsw : cur.switch = fun k j ↦ fam4 f00 f01 f10 f11 k j ⟨i, by omega⟩) :
    Rel sm nxt.key (Tc + lutBranchTargetG (q := q) (p := p) (sig tab W) hodd μ sp
        (fam4 f00 f01 f10 f11) i) nxt.row (Bs + lutBranchBoundG n m bound p B Bh) ∧
      nxt.switch = fun k j ↦ fam4 f00 f01 f10 f11 k j ⟨i + 1, hi⟩ := by
  subst hmw
  refine Exists.elim h (fun v0 h ↦ Exists.elim h (fun v1 h ↦ Exists.elim h (fun v2 h ↦ Exists.elim h (fun v3 h ↦ Exists.elim h (fun v4 h ↦ Exists.elim h (fun v5 h ↦ Exists.elim h (fun v6 h ↦ Exists.elim h (fun v7 h ↦ Exists.elim h (fun v8 h ↦ Exists.elim h (fun v9 h ↦ Exists.elim h (fun v10 h ↦ Exists.elim h (fun v11 h ↦ Exists.elim h (fun v12 h ↦ Exists.elim h (fun v13 h ↦ Exists.elim h (fun v14 h ↦ Exists.elim h (fun v15 h ↦ Exists.elim h (fun v16 h ↦ Exists.elim h (fun v17 h ↦ Exists.elim h (fun v18 h ↦ Exists.elim h (fun v19 h ↦ Exists.elim h (fun v20 h ↦ Exists.elim h (fun v21 h ↦ Exists.elim h (fun v22 h ↦ Exists.elim h (fun v23 h ↦ Exists.elim h (fun v24 h ↦ Exists.elim h (fun v25 h ↦ Exists.elim h (fun v26 h ↦ Exists.elim h (fun v27 h ↦ Exists.elim h (fun v28 h ↦ Exists.elim h (fun v29 h ↦ Exists.elim h (fun v30 h ↦ Exists.elim h (fun v31 h ↦ Exists.elim h (fun v32 h ↦ Exists.elim h (fun v33 h ↦ Exists.elim h (fun v34 h ↦ Exists.elim h (fun v35 h ↦ Exists.elim h (fun v36 h ↦ Exists.elim h (fun v37 h ↦ Exists.elim h (fun v38 h ↦ Exists.elim h (fun v39 h ↦ Exists.elim h (fun v40 h ↦ Exists.elim h (fun v41 h ↦ Exists.elim h (fun v42 h ↦ Exists.elim h (fun v43 h ↦ Exists.elim h (fun v44 h ↦ Exists.elim h (fun v45 h ↦ Exists.elim h (fun v46 h ↦ Exists.elim h (fun v47 h ↦ Exists.elim h (fun v48 h ↦ Exists.elim h (fun v49 h ↦ Exists.elim h (fun v50 h ↦ Exists.elim h (fun v51 h ↦ Exists.elim h (fun v52 h ↦ Exists.elim h (fun v53 h ↦ ?_))))))))))))))))))))))))))))))))))))))))))))))))))))))
  obtain ⟨-, -, a2, a3, a4, a5, a6, -, -, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19, a20, a21,
    a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38, a39, a40,
    a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, -, -, a56, -, -, a59, -, -,
    a62, -, -, a65, rfl⟩ := h
  obtain ⟨_, rfl⟩ := familyGet_lit i a2
  obtain ⟨_, rfl⟩ := familyGet_lit i a9
  obtain ⟨_, rfl⟩ := familyGet_succ i a56
  obtain ⟨_, rfl⟩ := familyGet_succ i a59
  obtain ⟨_, rfl⟩ := familyGet_succ i a62
  obtain ⟨_, rfl⟩ := familyGet_succ i a65
  obtain ⟨rfl, rfl, rfl, rfl⟩ := lutA_of_samples a3 a10 a4 a11
  refine lut_step_gen hn hpow hq hq1 hq' layout hlayout hexact (sig tab W) hodd hsm hg' hg hinput
    (fam4 f00 f01 f10 f11) i hi hcur hsw (hhelp ⟨i, by omega⟩) a15 a16 a17 a20 a21 a22 a18 a19
    a23 a24 a25 (Xb := ![v22, v34]) (Fin.forall_fin_two.2 ⟨a26, a38⟩)
    (D' := ![v23, v35]) (Fin.forall_fin_two.2 ⟨a27, a39⟩)
    (D := ![v24, v36]) (Fin.forall_fin_two.2 ⟨a28, a40⟩)
    (M := ![![v25, v37], ![v27, v39]]) (Fin.forall_fin_two.2 ⟨a29, a41⟩)
    (Fin.forall_fin_two.2 ⟨a31, a43⟩)
    (Asw := ![![v26, v38], ![v28, v40]])
    (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨a30, a42⟩, Fin.forall_fin_two.2 ⟨a32, a44⟩⟩)
    (Z := ![v29, v41]) (Fin.forall_fin_two.2 ⟨a33, a45⟩)
    (E' := ![v30, v42]) (Fin.forall_fin_two.2 ⟨a34, a46⟩)
    (E := ![v31, v43]) (Fin.forall_fin_two.2 ⟨a35, a47⟩)
    a5 a6 a12 a13 a14
    (Y := ![v32, v44]) (Fin.forall_fin_two.2 ⟨a36, a48⟩)
    (X := ![v33, v45]) (Fin.forall_fin_two.2 ⟨a37, a49⟩)
    (K := ![![v46, v47], ![v48, v49]]) (Fin.forall_fin_two.2 ⟨a50, a51⟩)
    (Fin.forall_fin_two.2 ⟨a52, a53⟩)
    (Fin.forall_fin_two.2 ⟨rfl, rfl⟩)
    (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨rfl, rfl⟩, Fin.forall_fin_two.2 ⟨rfl, rfl⟩⟩) ?_
  funext k j
  fin_cases k <;> fin_cases j <;> rfl

namespace BrLut

variable {n : Nat}

/-- The input tuple of a LUT branch-loop body: the state, the helper families, the hash key, the
LUT input `(c0, A00, A10, c1, A01, A11)` and the four switching families. -/
abbrev StepIn (n q q' m m' W : Nat) :=
  ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
    ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q' n 1 m' × ExactMatrix q' n 1 m' ×
    ExactMatrix q' n 1 m' × ExactMatrix q' n 1 m' × (Fin W → ExactMatrix q n 1 m) × ByteArray ×
    (Fin W → ExactMatrix q n 1 m) × ExactMatrix q n 1 m × ExactMatrix q n 1 m ×
    ExactMatrix q n 1 m × ExactMatrix q n 1 m × ExactMatrix q n 1 m × ExactMatrix q n 1 m ×
    (Fin W → ExactMatrix q' n 1 m') × (Fin W → ExactMatrix q' n 1 m') ×
    (Fin W → ExactMatrix q' n 1 m') × (Fin W → ExactMatrix q' n 1 m') × Unit

/-- The body input of iteration `i` on the state `cur`. -/
abbrev stepIn {q q' m m' W : Nat} (cur : LoopStateG n q q' m') (hf0 : Fin W → ExactMatrix q n 1 m)
    (key : ByteArray) (hf1 : Fin W → ExactMatrix q n 1 m) (x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m)
    (f00 f01 f10 f11 : Fin W → ExactMatrix q' n 1 m') : StepIn n q q' m m' W :=
  (cur.1, cur.2.1, cur.2.2.1, cur.2.2.2.1, cur.2.2.2.2.1, cur.2.2.2.2.2.1, cur.2.2.2.2.2.2.1,
    cur.2.2.2.2.2.2.2.1, cur.2.2.2.2.2.2.2.2.1, cur.2.2.2.2.2.2.2.2.2.1, hf0, key, hf1,
    x0, x1, x2, x3, x4, x5, f00, f01, f10, f11, ())

/-- **The step hypothesis of `lut_loop_gen`** for a branch loop with body `child`: iteration `i`
adds `T i` with bound `Bb` and moves the carried switching ciphertexts from branch `i` to `i + 1`. -/
def LutStepHyp {q q' m m' W : Nat} (child : Nat → StepIn n q q' m m' W → LoopStateG n q q' m' → Prop)
    (hf0 : Fin W → ExactMatrix q n 1 m) (key : ByteArray) (hf1 : Fin W → ExactMatrix q n 1 m)
    (x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m) (f00 f01 f10 f11 : Fin W → ExactMatrix q' n 1 m')
    (sm : ErrorPoly n) (T : Nat → Fin 2 → ExactMatrix q n 1 m') (Bb : Nat) : Prop :=
  ∀ i (hi : i + 1 < W) (cur nxt : LoopStateG n q q' m') (Tc : Fin 2 → ExactMatrix q n 1 m')
    (Bs : Nat), child i (stepIn cur hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11) nxt →
    cur.switch = (fun k j ↦ fam4 f00 f01 f10 f11 k j ⟨i, by omega⟩) →
    Rel sm cur.key Tc cur.row Bs →
    Rel sm nxt.key (Tc + T i) nxt.row (Bs + Bb) ∧
      nxt.switch = fun k j ↦ fam4 f00 f01 f10 f11 k j ⟨i + 1, hi⟩

/-- A branch loop started from zero sums on member `0` of the switching families: after `L`
iterations it encodes the first `L` branches with bound `L Bb` and carries branch `L`. -/
theorem loop_of {q q' m m' W : Nat} {child : Nat → StepIn n q q' m m' W → LoopStateG n q q' m' → Prop}
    {hf0 : Fin W → ExactMatrix q n 1 m} {key : ByteArray} {hf1 : Fin W → ExactMatrix q n 1 m}
    {x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m} {f00 f01 f10 f11 : Fin W → ExactMatrix q' n 1 m'}
    {sm : ErrorPoly n} {T : Nat → Fin 2 → ExactMatrix q n 1 m'} {Bb : Nat}
    (hchild : LutStepHyp child hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11 sm T Bb)
    {L : Nat} (hL : L < W) {v0 v1 v2 v3 : ExactMatrix q' n 1 m'}
    (h0 : familyGetStatic f00 0 v0) (h1 : familyGetStatic f01 0 v1)
    (h2 : familyGetStatic f10 0 v2) (h3 : familyGetStatic f11 0 v3) {F : LoopStateG n q q' m'}
    (hruns : MxxIR.IterRuns (fun i current next ↦
      child i (stepIn current hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11) next) L
      ((0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'),
        (0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'),
        v0, v1, v2, v3, ()) F) :
    Rel sm F.key (∑ i ∈ Finset.range L, T i) F.row (L * Bb) ∧
      F.switch = fun k j ↦ fam4 f00 f01 f10 f11 k j ⟨L, hL⟩ := by
  obtain ⟨_, rfl⟩ := familyGet_lit 0 h0
  obtain ⟨_, rfl⟩ := familyGet_lit 0 h1
  obtain ⟨_, rfl⟩ := familyGet_lit 0 h2
  obtain ⟨_, rfl⟩ := familyGet_lit 0 h3
  refine lut_loop_gen LoopStateG.key LoopStateG.row LoopStateG.switch T (fam4 f00 f01 f10 f11) Bb
    hL ?_ ?_ (fun i hi cur nxt Tc Bs hb hsw hrel ↦ hchild i hi cur nxt Tc Bs hb hsw hrel) hruns
  · have hk : LoopStateG.key ((0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'),
        (0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'),
        (0 : ExactMatrix q n 1 m'), f00 ⟨0, by omega⟩, f01 ⟨0, by omega⟩, f10 ⟨0, by omega⟩,
        f11 ⟨0, by omega⟩, ()) = 0 := by
      funext k j; fin_cases k <;> fin_cases j <;> rfl
    have hr : LoopStateG.row ((0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'),
        (0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'), (0 : ExactMatrix q n 1 m'),
        (0 : ExactMatrix q n 1 m'), f00 ⟨0, by omega⟩, f01 ⟨0, by omega⟩, f10 ⟨0, by omega⟩,
        f11 ⟨0, by omega⟩, ()) = 0 := by
      funext j; fin_cases j <;> rfl
    rw [hk, hr]
    exact Rel.zero
  · funext k j; fin_cases k <;> fin_cases j <;> rfl

/-- An encoding tuple `(A00, A01, A10, A11, c0, c1)` from a relation on any functions equal to
its blocks. -/
theorem encRel_of {q m : Nat} {sm : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {t c : Fin 2 → ExactMatrix q n 1 m} {B : Nat} (h : Rel sm A t c B)
    {a00 a01 a10 a11 c0 c1 : ExactMatrix q n 1 m}
    (hA : ∀ k j, A k j = ![![a00, a01], ![a10, a11]] k j) (hc : ∀ j, c j = ![c0, c1] j) :
    EncRel sm (a00, a01, a10, a11, c0, c1, ()) t B := by
  have e1 : A = Enc.A (a00, a01, a10, a11, c0, c1, ()) := by
    funext k j; rw [hA]; fin_cases k <;> fin_cases j <;> rfl
  have e2 : c = Enc.c (a00, a01, a10, a11, c0, c1, ()) := by
    funext j; rw [hc]; fin_cases j <;> rfl
  rw [e1, e2] at h
  exact h

/-- The weight entry of `matrixPolynomial [w]`. -/
theorem matrixPolynomial_entry {q : Nat} (w : Int) :
    (matrixPolynomial [w] : ExactMatrix q n 1 1) 0 0 = ((w : Int) : ExactPoly q n) := by
  simp [matrixPolynomial]

end BrLut

/-! ## The keys of one iteration (the LUT key link) -/

set_option maxHeartbeats 64000000 in
/-- **The keys of one iteration of a LUT branch loop** are the preprocessing's branch keys:
`nxt.key = cur.key + branchKeyG … σ_i A (lutAG … σ_i) cur.switch` (`branchKeyG_of_stacks`), with
`A` the LUT input keys; with `carried_loop_gen` and `lut_keys_link_gen` this is the LUT key link. -/
theorem LutRoundStep.keys {n q q' m m' W : Nat} {backend : BackendContext} {hashModel : HashModel}
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
        x0, x1, x2, x3, x4, x5, f00, f01, f10, f11, ()) nxt)
    (hmw : mw = (m' : Int)) (hn : 0 < n) (hq' : 1 < q') (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    (hm : m = 1 * layout.digitCount) :
    nxt.key = cur.key + branchKeyG backend base digits basis
      (MxxIR.selectInt (Int.ofNat i) tab).toNat (inA x0 x1 x2 x3 x4 x5)
      (lutAG hashModel key q m tag level (MxxIR.selectInt (Int.ofNat i) tab)) cur.switch := by
  subst hmw
  refine Exists.elim h (fun v0 h ↦ Exists.elim h (fun v1 h ↦ Exists.elim h (fun v2 h ↦ Exists.elim h (fun v3 h ↦ Exists.elim h (fun v4 h ↦ Exists.elim h (fun v5 h ↦ Exists.elim h (fun v6 h ↦ Exists.elim h (fun v7 h ↦ Exists.elim h (fun v8 h ↦ Exists.elim h (fun v9 h ↦ Exists.elim h (fun v10 h ↦ Exists.elim h (fun v11 h ↦ Exists.elim h (fun v12 h ↦ Exists.elim h (fun v13 h ↦ Exists.elim h (fun v14 h ↦ Exists.elim h (fun v15 h ↦ Exists.elim h (fun v16 h ↦ Exists.elim h (fun v17 h ↦ Exists.elim h (fun v18 h ↦ Exists.elim h (fun v19 h ↦ Exists.elim h (fun v20 h ↦ Exists.elim h (fun v21 h ↦ Exists.elim h (fun v22 h ↦ Exists.elim h (fun v23 h ↦ Exists.elim h (fun v24 h ↦ Exists.elim h (fun v25 h ↦ Exists.elim h (fun v26 h ↦ Exists.elim h (fun v27 h ↦ Exists.elim h (fun v28 h ↦ Exists.elim h (fun v29 h ↦ Exists.elim h (fun v30 h ↦ Exists.elim h (fun v31 h ↦ Exists.elim h (fun v32 h ↦ Exists.elim h (fun v33 h ↦ Exists.elim h (fun v34 h ↦ Exists.elim h (fun v35 h ↦ Exists.elim h (fun v36 h ↦ Exists.elim h (fun v37 h ↦ Exists.elim h (fun v38 h ↦ Exists.elim h (fun v39 h ↦ Exists.elim h (fun v40 h ↦ Exists.elim h (fun v41 h ↦ Exists.elim h (fun v42 h ↦ Exists.elim h (fun v43 h ↦ Exists.elim h (fun v44 h ↦ Exists.elim h (fun v45 h ↦ Exists.elim h (fun v46 h ↦ Exists.elim h (fun v47 h ↦ Exists.elim h (fun v48 h ↦ Exists.elim h (fun v49 h ↦ Exists.elim h (fun v50 h ↦ Exists.elim h (fun v51 h ↦ Exists.elim h (fun v52 h ↦ Exists.elim h (fun v53 h ↦ ?_))))))))))))))))))))))))))))))))))))))))))))))))))))))
  obtain ⟨-, -, a2, a3, a4, a5, a6, -, -, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
    a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38,
    a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, -, -, -, -, -, -,
    -, -, -, -, -, -, rfl⟩ := h
  obtain ⟨rfl, rfl, rfl, rfl⟩ := lutA_of_samples a3 a10 a4 a11
  have hK := branchKeyG_of_stacks (c := inC x0 x1 x2 x3 x4 x5) (A := inA x0 x1 x2 x3 x4 x5)
    hn hq' layout hlayout hexact hm a15 a16 a17 a20 a21 a22 a18 a19 a23 a24 a25
    (sw := cur.switch) (Xb := ![v22, v34]) (Fin.forall_fin_two.2 ⟨a26, a38⟩)
    (D' := ![v23, v35]) (Fin.forall_fin_two.2 ⟨a27, a39⟩)
    (D := ![v24, v36]) (Fin.forall_fin_two.2 ⟨a28, a40⟩)
    (M := ![![v25, v37], ![v27, v39]]) (Fin.forall_fin_two.2 ⟨a29, a41⟩)
    (Fin.forall_fin_two.2 ⟨a31, a43⟩) (Asw := ![![v26, v38], ![v28, v40]])
    (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨a30, a42⟩, Fin.forall_fin_two.2 ⟨a32, a44⟩⟩)
    (Z := ![v29, v41]) (Fin.forall_fin_two.2 ⟨a33, a45⟩)
    (E' := ![v30, v42]) (Fin.forall_fin_two.2 ⟨a34, a46⟩)
    (E := ![v31, v43]) (Fin.forall_fin_two.2 ⟨a35, a47⟩)
    (h := ![v0, v5]) a5 a6 a12 a13 a14
    (K := ![![v46, v47], ![v48, v49]]) (Fin.forall_fin_two.2 ⟨a50, a51⟩)
    (Fin.forall_fin_two.2 ⟨a52, a53⟩)
  rw [← hK]
  funext k j
  fin_cases k <;> fin_cases j <;> rfl

/-- **The step hypothesis of every iteration** of a branch loop whose body `child` bundles to
`LutRoundStep.Comp` (`hcomp`: the child scope's adapter): `LutRoundStep.rel` for each `i`. -/
theorem LutRoundStep.hstep {n q q' m m' W : Nat} {backend : BackendContext} {hashModel : HashModel}
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
    (hmw : mw = (m' : Int)) (hn : 0 < n) {log : Nat} (hpow : n = 2 ^ log)
    {p : Nat} (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q')
    (layout : RegularLayout q') (hlayout : backend.regularLayout q' n = some layout)
    (hexact : layout.droppedModuli = 0) (hodd : ∀ k, Odd (sig tab W k).toNat)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix q' n 1 m} {g : ExactMatrix q n 1 m}
    (hg' : gadgetMatrixRuns backend base digits g') (hg : centeredRebaseRuns g' g) {B : Nat}
    (hinput : Rel sm (inA x0 x1 x2 x3 x4 x5)
      (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) (inC x0 x1 x2 x3 x4 x5) B)
    {Bh : Nat}
    (hhelp : ∀ k : Fin W, Rel sm (lutAG hashModel key q m tag level (sig tab W k))
      (fun j ↦ reducePoly q n ((p : Int) *
        unitPair (substitution (sig tab W k).toNat (hodd k) sm) j) • g) ![hf0 k, hf1 k] Bh) :
    LutStepHyp child hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11 sm
      (lutBranchTargetG (q := q) (p := p) (sig tab W) hodd μ sp (fam4 f00 f01 f10 f11))
      (lutBranchBoundG n m bound p B Bh) :=
  fun i hi cur nxt _ _ hc hsw hcur ↦ LutRoundStep.rel (hcomp i cur nxt hc) hmw hn hpow hq hq1 hq'
    layout hlayout hexact hodd hsm hg' hg hinput hhelp hi hcur hsw

/-! ## LUT members: readout, sampler, sampler with offset, smudge -/

set_option maxHeartbeats 64000000 in
/-- **A readout LUT member** (`lut.member.readout`, a width-`W` LUT: `L = W - 1` looped branches, the last
branch and the move down to `q'`): with `hchild` (from `LutRoundStep.hstep` and the child adapter)
and switching ciphertexts encoding the coefficients `C` (`hswitch`) with `Σ_k σ_k(μ) C_k = ν`
(`hwalsh`), the output encodes `Δ' ν` at `q'` against `gq` (`lut_round_gen`). -/
theorem LutMemberReadout.rel {n q q' m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {tab : List Int} {key : ByteArray} {level : Int}
    {tag : Blob} {hf0 hf1 : Fin W → ExactMatrix q n 1 m} {x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m}
    {f00 f01 f10 f11 : Fin W → ExactMatrix q' n 1 m'}
    {child : Nat → StepIn n q q' m m' W → LoopStateG n q q' m' → Prop} {i : Nat} {ℓ : Int}
    {bound : Nat} {basis basis' : List Nat} {mw : Int}
    {outputs : ExactMatrix q' n 1 m' × ExactMatrix q' n 1 m' × ExactMatrix q' n 1 m' ×
      ExactMatrix q' n 1 m' × ExactMatrix q' n 1 m' × ExactMatrix q' n 1 m' × Unit}
    (h : LutMemberReadout.Comp q n m' q' m' q' m' W m W m m m W W W W m' m' q' q' q' q' q' q' q' q'
      0 0 0 0 ℓ ℓ level tag ℓ tab level tag ℓ tab ℓ level tag ℓ tab level tag ℓ tab
      ℓ tab ℓ tab ℓ tab ℓ tab ℓ tab ℓ tab
      base digits bound mw basis mw basis base digits bound mw basis' base digits bound mw basis mw
      basis base digits bound mw basis' mw basis' mw basis' mw mw basis' mw mw basis'
      backend hashModel i child (f00, f01, f10, f11, hf0, key, hf1, x0, x1, x2, x3, x4, x5, ())
      outputs)
    (hmw : mw = (m' : Int)) (L : Nat) (hℓ : ℓ = (L : Int)) (hL : L + 1 = W) (hn : 0 < n) {log : Nat} (hpow : n = 2 ^ log)
    {p : Nat} (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p)
    (layout : RegularLayout q') (hlayout : backend.regularLayout q' n = some layout)
    (hexact : layout.droppedModuli = 0) (hodd : ∀ k, Odd (sig tab W k).toNat)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix q' n 1 m} {g : ExactMatrix q n 1 m}
    (hg' : gadgetMatrixRuns backend base digits g') (hg : centeredRebaseRuns g' g) {B : Nat}
    (hinput : Rel sm (inA x0 x1 x2 x3 x4 x5)
      (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) (inC x0 x1 x2 x3 x4 x5) B)
    {Bh : Nat}
    (hhelp : ∀ k : Fin W, Rel sm (lutAG hashModel key q m tag level (sig tab W k))
      (fun j ↦ reducePoly q n ((p : Int) *
        unitPair (substitution (sig tab W k).toNat (hodd k) sm) j) • g) ![hf0 k, hf1 k] Bh)
    (hchild : LutStepHyp child hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11 sm
      (lutBranchTargetG (q := q) (p := p) (sig tab W) hodd μ sp (fam4 f00 f01 f10 f11))
      (lutBranchBoundG n m bound p B Bh))
    {Δ' ν : ErrorPoly n} {gq : ExactMatrix q' n 1 m'} {C : Fin W → ExactPoly q' n} {BB : Nat}
    (hswitch : ∀ k : Fin W, ∃ e : Fin 2 → ErrorMatrix n 1 m', (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly q' n (substitution (sig tab W k).toNat (hodd k) sp) •
          fam4 f00 f01 f10 f11 0 j k + fam4 f00 f01 f10 f11 1 j k =
        (C k * reducePoly q' n (Δ' * unitPair sp j)) • gq + reduceMatrix q' n 1 m' (e j))
    (hμ : ∀ k, polyNorm (substitution (sig tab W k).toNat (hodd k) μ) ≤ 1)
    (hwalsh : ∑ k, reducePoly q' n (substitution (sig tab W k).toNat (hodd k) μ) * C k =
      reducePoly q' n ν) :
    EncRel sm outputs (fun j ↦ reducePoly q' n (Δ' * ν * unitPair sp j) • gq)
      ((L * lutBranchBoundG n m bound p B Bh + lutBranchBoundG n m bound p B Bh + (n + 2) * (p / 2))
        / p + W * (n * 1 * BB)) := by
  subst hmw hℓ
  refine Exists.elim h (fun v0 h ↦ Exists.elim h (fun v1 h ↦ Exists.elim h (fun v2 h ↦ Exists.elim h (fun v3 h ↦ Exists.elim h (fun v4 h ↦ Exists.elim h (fun v5 h ↦ Exists.elim h (fun v6 h ↦ Exists.elim h (fun v7 h ↦ Exists.elim h (fun v8 h ↦ Exists.elim h (fun v9 h ↦ Exists.elim h (fun v10 h ↦ Exists.elim h (fun v11 h ↦ Exists.elim h (fun v12 h ↦ Exists.elim h (fun v13 h ↦ Exists.elim h (fun v14 h ↦ Exists.elim h (fun v15 h ↦ Exists.elim h (fun v16 h ↦ Exists.elim h (fun v17 h ↦ Exists.elim h (fun v18 h ↦ Exists.elim h (fun v19 h ↦ Exists.elim h (fun v20 h ↦ Exists.elim h (fun v21 h ↦ Exists.elim h (fun v22 h ↦ Exists.elim h (fun v23 h ↦ Exists.elim h (fun v24 h ↦ Exists.elim h (fun v25 h ↦ Exists.elim h (fun v26 h ↦ Exists.elim h (fun v27 h ↦ Exists.elim h (fun v28 h ↦ Exists.elim h (fun v29 h ↦ Exists.elim h (fun v30 h ↦ Exists.elim h (fun v31 h ↦ Exists.elim h (fun v32 h ↦ Exists.elim h (fun v33 h ↦ Exists.elim h (fun v34 h ↦ Exists.elim h (fun v35 h ↦ Exists.elim h (fun v36 h ↦ Exists.elim h (fun v37 h ↦ Exists.elim h (fun v38 h ↦ Exists.elim h (fun v39 h ↦ Exists.elim h (fun v40 h ↦ Exists.elim h (fun v41 h ↦ Exists.elim h (fun v42 h ↦ Exists.elim h (fun v43 h ↦ Exists.elim h (fun v44 h ↦ Exists.elim h (fun v45 h ↦ Exists.elim h (fun v46 h ↦ Exists.elim h (fun v47 h ↦ Exists.elim h (fun v48 h ↦ Exists.elim h (fun v49 h ↦ Exists.elim h (fun v50 h ↦ Exists.elim h (fun v51 h ↦ Exists.elim h (fun v52 h ↦ Exists.elim h (fun v53 h ↦ Exists.elim h (fun v54 h ↦ Exists.elim h (fun v55 h ↦ Exists.elim h (fun v56 h ↦ Exists.elim h (fun v57 h ↦ Exists.elim h (fun v58 h ↦ Exists.elim h (fun v59 h ↦ Exists.elim h (fun v60 h ↦ Exists.elim h (fun v61 h ↦ Exists.elim h (fun v62 h ↦ Exists.elim h (fun v63 h ↦ Exists.elim h (fun v64 h ↦ Exists.elim h (fun v65 h ↦ Exists.elim h (fun v66 h ↦ Exists.elim h (fun v67 h ↦ Exists.elim h (fun v68 h ↦ Exists.elim h (fun v69 h ↦ ?_))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
    a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38,
    a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56, a57,
    a58, a59, a60, rfl⟩ := h
  obtain ⟨hloop, hsw⟩ := loop_of hchild (L := L) (by omega) a0 a1 a2 a3 a4
  obtain ⟨_, rfl⟩ := familyGet_lit L a5
  obtain ⟨_, rfl⟩ := familyGet_lit L a10
  obtain ⟨rfl, rfl, rfl, rfl⟩ := lutA_of_samples a6 a11 a7 a12
  exact encRel_of (lut_round_gen hn hpow (L := L) hL hq hq1 hq' hp layout hlayout hexact
    (sig tab W) hodd hsm hg' hg hinput hloop hsw (hhelp ⟨L, by omega⟩)
    a16 a17 a18 a21 a22 a23 a19 a20 a24 a25 a26
    (Xb := ![v36, v48]) (Fin.forall_fin_two.2 ⟨a27, a39⟩)
    (D' := ![v37, v49]) (Fin.forall_fin_two.2 ⟨a28, a40⟩)
    (D := ![v38, v50]) (Fin.forall_fin_two.2 ⟨a29, a41⟩)
    (M := ![![v39, v51], ![v41, v53]]) (Fin.forall_fin_two.2 ⟨a30, a42⟩)
    (Fin.forall_fin_two.2 ⟨a32, a44⟩) (Asw := ![![v40, v52], ![v42, v54]])
    (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨a31, a43⟩, Fin.forall_fin_two.2 ⟨a33, a45⟩⟩)
    (Z := ![v43, v55]) (Fin.forall_fin_two.2 ⟨a34, a46⟩)
    (E' := ![v44, v56]) (Fin.forall_fin_two.2 ⟨a35, a47⟩)
    (E := ![v45, v57]) (Fin.forall_fin_two.2 ⟨a36, a48⟩)
    a8 a9 a13 a14 a15
    (Y := ![v64, v67]) (Fin.forall_fin_two.2 ⟨a55, a58⟩)
    (X := ![v65, v68]) (Fin.forall_fin_two.2 ⟨a56, a59⟩)
    (K := ![![v46, v58], ![v60, v62]]) (Fin.forall_fin_two.2 ⟨a37, a49⟩)
    (Fin.forall_fin_two.2 ⟨a51, a53⟩) hswitch hμ hwalsh
    (K' := ![![v47, v59], ![v61, v63]])
    (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨a38, a50⟩, Fin.forall_fin_two.2 ⟨a52, a54⟩⟩)
    (R' := ![v66, v69]) (Fin.forall_fin_two.2 ⟨a57, a60⟩))
    (fun _ _ ↦ rfl) (fun _ ↦ rfl)

set_option maxHeartbeats 64000000 in
/-- **A sampling LUT member** (`lut.member.sampler`, width `W = L + 1`, no move down): the output encodes
the sum of the `W` branches on the input `μ` (`lut_branches_gen`), the `hmem` of
`sampler_carrier_gen` up to the offset. -/
theorem LutMemberSampler.rel {n q q' m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {tab : List Int} {key : ByteArray} {level : Int}
    {tag : Blob} {hf0 hf1 : Fin W → ExactMatrix q n 1 m} {x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m}
    {f00 f01 f10 f11 : Fin W → ExactMatrix q' n 1 m'}
    {child : Nat → StepIn n q q' m m' W → LoopStateG n q q' m' → Prop} {i : Nat} {ℓ : Int}
    {bound : Nat} {basis : List Nat} {mw : Int}
    {outputs : ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × Unit}
    (h : LutMemberSampler.Comp q n m' q' m' q' m' W m W m m m W W W W m' m' q' q'
      0 0 0 0 ℓ ℓ level tag ℓ tab level tag ℓ tab ℓ level tag ℓ tab level tag ℓ tab
      ℓ tab ℓ tab ℓ tab ℓ tab ℓ tab ℓ tab
      base digits bound mw basis mw basis base digits bound mw base digits bound mw basis mw basis
      base digits bound mw mw mw mw mw mw mw
      backend hashModel i child (f00, f01, f10, f11, hf0, key, hf1, x0, x1, x2, x3, x4, x5, ())
      outputs)
    (hmw : mw = (m' : Int)) (L : Nat) (hℓ : ℓ = (L : Int)) (hL : L + 1 = W) (hn : 0 < n) {log : Nat} (hpow : n = 2 ^ log)
    {p : Nat} (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q')
    (layout : RegularLayout q') (hlayout : backend.regularLayout q' n = some layout)
    (hexact : layout.droppedModuli = 0) (hodd : ∀ k, Odd (sig tab W k).toNat)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix q' n 1 m} {g : ExactMatrix q n 1 m}
    (hg' : gadgetMatrixRuns backend base digits g') (hg : centeredRebaseRuns g' g) {B : Nat}
    (hinput : Rel sm (inA x0 x1 x2 x3 x4 x5)
      (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) (inC x0 x1 x2 x3 x4 x5) B)
    {Bh : Nat}
    (hhelp : ∀ k : Fin W, Rel sm (lutAG hashModel key q m tag level (sig tab W k))
      (fun j ↦ reducePoly q n ((p : Int) *
        unitPair (substitution (sig tab W k).toNat (hodd k) sm) j) • g) ![hf0 k, hf1 k] Bh)
    (hchild : LutStepHyp child hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11 sm
      (lutBranchTargetG (q := q) (p := p) (sig tab W) hodd μ sp (fam4 f00 f01 f10 f11))
      (lutBranchBoundG n m bound p B Bh)) :
    EncRel sm outputs (fun j ↦ ∑ k : Fin W, branchTargetG (q := q) (p := p) (sig tab W) hodd μ sp
        (fam4 f00 f01 f10 f11) k j)
      (L * lutBranchBoundG n m bound p B Bh + lutBranchBoundG n m bound p B Bh) := by
  subst hmw hℓ
  refine Exists.elim h (fun v0 h ↦ Exists.elim h (fun v1 h ↦ Exists.elim h (fun v2 h ↦ Exists.elim h (fun v3 h ↦ Exists.elim h (fun v4 h ↦ Exists.elim h (fun v5 h ↦ Exists.elim h (fun v6 h ↦ Exists.elim h (fun v7 h ↦ Exists.elim h (fun v8 h ↦ Exists.elim h (fun v9 h ↦ Exists.elim h (fun v10 h ↦ Exists.elim h (fun v11 h ↦ Exists.elim h (fun v12 h ↦ Exists.elim h (fun v13 h ↦ Exists.elim h (fun v14 h ↦ Exists.elim h (fun v15 h ↦ Exists.elim h (fun v16 h ↦ Exists.elim h (fun v17 h ↦ Exists.elim h (fun v18 h ↦ Exists.elim h (fun v19 h ↦ Exists.elim h (fun v20 h ↦ Exists.elim h (fun v21 h ↦ Exists.elim h (fun v22 h ↦ Exists.elim h (fun v23 h ↦ Exists.elim h (fun v24 h ↦ Exists.elim h (fun v25 h ↦ Exists.elim h (fun v26 h ↦ Exists.elim h (fun v27 h ↦ Exists.elim h (fun v28 h ↦ Exists.elim h (fun v29 h ↦ Exists.elim h (fun v30 h ↦ Exists.elim h (fun v31 h ↦ Exists.elim h (fun v32 h ↦ Exists.elim h (fun v33 h ↦ Exists.elim h (fun v34 h ↦ Exists.elim h (fun v35 h ↦ Exists.elim h (fun v36 h ↦ Exists.elim h (fun v37 h ↦ Exists.elim h (fun v38 h ↦ Exists.elim h (fun v39 h ↦ Exists.elim h (fun v40 h ↦ Exists.elim h (fun v41 h ↦ Exists.elim h (fun v42 h ↦ Exists.elim h (fun v43 h ↦ Exists.elim h (fun v44 h ↦ Exists.elim h (fun v45 h ↦ Exists.elim h (fun v46 h ↦ Exists.elim h (fun v47 h ↦ Exists.elim h (fun v48 h ↦ Exists.elim h (fun v49 h ↦ Exists.elim h (fun v50 h ↦ Exists.elim h (fun v51 h ↦ Exists.elim h (fun v52 h ↦ Exists.elim h (fun v53 h ↦ Exists.elim h (fun v54 h ↦ Exists.elim h (fun v55 h ↦ Exists.elim h (fun v56 h ↦ Exists.elim h (fun v57 h ↦ Exists.elim h (fun v58 h ↦ Exists.elim h (fun v59 h ↦ Exists.elim h (fun v60 h ↦ Exists.elim h (fun v61 h ↦ Exists.elim h (fun v62 h ↦ Exists.elim h (fun v63 h ↦ ?_))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
    a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38,
    a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, rfl⟩ := h
  obtain ⟨hloop, hsw⟩ := loop_of hchild (L := L) (by omega) a0 a1 a2 a3 a4
  obtain ⟨_, rfl⟩ := familyGet_lit L a5
  obtain ⟨_, rfl⟩ := familyGet_lit L a10
  obtain ⟨rfl, rfl, rfl, rfl⟩ := lutA_of_samples a6 a11 a7 a12
  exact encRel_of (lut_branches_gen hn hpow (L := L) hL hq hq1 hq' layout hlayout
    hexact (sig tab W) hodd hsm hg' hg hinput hloop hsw (hhelp ⟨L, by omega⟩)
    a16 a17 a18 a21 a22 a23 a19 a20 a24 a25 a26
    (Xb := ![v36, v47]) (Fin.forall_fin_two.2 ⟨a27, a38⟩)
    (D' := ![v37, v48]) (Fin.forall_fin_two.2 ⟨a28, a39⟩)
    (D := ![v38, v49]) (Fin.forall_fin_two.2 ⟨a29, a40⟩)
    (M := ![![v39, v50], ![v41, v52]]) (Fin.forall_fin_two.2 ⟨a30, a41⟩)
    (Fin.forall_fin_two.2 ⟨a32, a43⟩) (Asw := ![![v40, v51], ![v42, v53]])
    (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨a31, a42⟩, Fin.forall_fin_two.2 ⟨a33, a44⟩⟩)
    (Z := ![v43, v54]) (Fin.forall_fin_two.2 ⟨a34, a45⟩)
    (E' := ![v44, v55]) (Fin.forall_fin_two.2 ⟨a35, a46⟩)
    (E := ![v45, v56]) (Fin.forall_fin_two.2 ⟨a36, a47⟩)
    a8 a9 a13 a14 a15
    (Y := ![v60, v62]) (Fin.forall_fin_two.2 ⟨a51, a53⟩)
    (X := ![v61, v63]) (Fin.forall_fin_two.2 ⟨a52, a54⟩)
    (K := ![![v46, v57], ![v58, v59]]) (Fin.forall_fin_two.2 ⟨a37, a48⟩)
    (Fin.forall_fin_two.2 ⟨a49, a50⟩))
    (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨rfl, rfl⟩, Fin.forall_fin_two.2 ⟨rfl, rfl⟩⟩)
    (Fin.forall_fin_two.2 ⟨rfl, rfl⟩)

set_option maxHeartbeats 64000000 in
/-- **A sampling LUT member with the offset** (`lut.member.sampler-offset`): the sampler's four
branches on `μ` plus the offset encoding `(o0, …, o5)` (layout `(A00, A01, A10, A11, c0, c1)`) of
`toff`: the `hmem` of `sampler_carrier_gen` with `toff` the offset branches. -/
theorem LutMemberSamplerOffset.rel {n q q' m m' W : Nat} {backend : BackendContext}
    {hashModel : HashModel} {base digits : Int} {tab : List Int} {key : ByteArray} {level : Int}
    {tag : Blob} {hf0 hf1 : Fin W → ExactMatrix q n 1 m} {x0 x1 x2 x3 x4 x5 : ExactMatrix q n 1 m}
    {o0 o1 o2 o3 o4 o5 : ExactMatrix q n 1 m'} {f00 f01 f10 f11 : Fin W → ExactMatrix q' n 1 m'}
    {child : Nat → StepIn n q q' m m' W → LoopStateG n q q' m' → Prop} {i : Nat} {ℓ : Int}
    {bound : Nat} {basis : List Nat} {mw : Int}
    {outputs : ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' ×
      ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × ExactMatrix q n 1 m' × Unit}
    (h : LutMemberSamplerOffset.Comp q n m' q' m' q' m' W m W m m m W W W W m' m' q' q'
      0 0 0 0 ℓ ℓ level tag ℓ tab level tag ℓ tab ℓ level tag ℓ tab level tag ℓ tab
      ℓ tab ℓ tab ℓ tab ℓ tab ℓ tab ℓ tab
      base digits bound mw basis mw basis base digits bound mw base digits bound mw basis mw basis
      base digits bound mw mw mw mw mw mw mw
      backend hashModel i child
      (f00, f01, f10, f11, hf0, key, hf1, x0, x1, x2, x3, x4, x5, o0, o1, o2, o3, o4, o5, ()) outputs)
    (hmw : mw = (m' : Int)) (L : Nat) (hℓ : ℓ = (L : Int)) (hL : L + 1 = W) (hn : 0 < n) {log : Nat} (hpow : n = 2 ^ log)
    {p : Nat} (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q')
    (layout : RegularLayout q') (hlayout : backend.regularLayout q' n = some layout)
    (hexact : layout.droppedModuli = 0) (hodd : ∀ k, Odd (sig tab W k).toNat)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix q' n 1 m} {g : ExactMatrix q n 1 m}
    (hg' : gadgetMatrixRuns backend base digits g') (hg : centeredRebaseRuns g' g) {B : Nat}
    (hinput : Rel sm (inA x0 x1 x2 x3 x4 x5)
      (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) (inC x0 x1 x2 x3 x4 x5) B)
    {Bh : Nat}
    (hhelp : ∀ k : Fin W, Rel sm (lutAG hashModel key q m tag level (sig tab W k))
      (fun j ↦ reducePoly q n ((p : Int) *
        unitPair (substitution (sig tab W k).toNat (hodd k) sm) j) • g) ![hf0 k, hf1 k] Bh)
    (hchild : LutStepHyp child hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11 sm
      (lutBranchTargetG (q := q) (p := p) (sig tab W) hodd μ sp (fam4 f00 f01 f10 f11))
      (lutBranchBoundG n m bound p B Bh))
    {toff : Fin 2 → ExactMatrix q n 1 m'} {Boff : Nat}
    (hoff : EncRel sm (o0, o1, o2, o3, o4, o5, ()) toff Boff) :
    EncRel sm outputs (fun j ↦ (∑ k : Fin W, branchTargetG (q := q) (p := p) (sig tab W) hodd μ sp
        (fam4 f00 f01 f10 f11) k j) + toff j)
      (L * lutBranchBoundG n m bound p B Bh + lutBranchBoundG n m bound p B Bh + Boff) := by
  subst hmw hℓ
  refine Exists.elim h (fun v0 h ↦ Exists.elim h (fun v1 h ↦ Exists.elim h (fun v2 h ↦ Exists.elim h (fun v3 h ↦ Exists.elim h (fun v4 h ↦ Exists.elim h (fun v5 h ↦ Exists.elim h (fun v6 h ↦ Exists.elim h (fun v7 h ↦ Exists.elim h (fun v8 h ↦ Exists.elim h (fun v9 h ↦ Exists.elim h (fun v10 h ↦ Exists.elim h (fun v11 h ↦ Exists.elim h (fun v12 h ↦ Exists.elim h (fun v13 h ↦ Exists.elim h (fun v14 h ↦ Exists.elim h (fun v15 h ↦ Exists.elim h (fun v16 h ↦ Exists.elim h (fun v17 h ↦ Exists.elim h (fun v18 h ↦ Exists.elim h (fun v19 h ↦ Exists.elim h (fun v20 h ↦ Exists.elim h (fun v21 h ↦ Exists.elim h (fun v22 h ↦ Exists.elim h (fun v23 h ↦ Exists.elim h (fun v24 h ↦ Exists.elim h (fun v25 h ↦ Exists.elim h (fun v26 h ↦ Exists.elim h (fun v27 h ↦ Exists.elim h (fun v28 h ↦ Exists.elim h (fun v29 h ↦ Exists.elim h (fun v30 h ↦ Exists.elim h (fun v31 h ↦ Exists.elim h (fun v32 h ↦ Exists.elim h (fun v33 h ↦ Exists.elim h (fun v34 h ↦ Exists.elim h (fun v35 h ↦ Exists.elim h (fun v36 h ↦ Exists.elim h (fun v37 h ↦ Exists.elim h (fun v38 h ↦ Exists.elim h (fun v39 h ↦ Exists.elim h (fun v40 h ↦ Exists.elim h (fun v41 h ↦ Exists.elim h (fun v42 h ↦ Exists.elim h (fun v43 h ↦ Exists.elim h (fun v44 h ↦ Exists.elim h (fun v45 h ↦ Exists.elim h (fun v46 h ↦ Exists.elim h (fun v47 h ↦ Exists.elim h (fun v48 h ↦ Exists.elim h (fun v49 h ↦ Exists.elim h (fun v50 h ↦ Exists.elim h (fun v51 h ↦ Exists.elim h (fun v52 h ↦ Exists.elim h (fun v53 h ↦ Exists.elim h (fun v54 h ↦ Exists.elim h (fun v55 h ↦ Exists.elim h (fun v56 h ↦ Exists.elim h (fun v57 h ↦ Exists.elim h (fun v58 h ↦ Exists.elim h (fun v59 h ↦ Exists.elim h (fun v60 h ↦ Exists.elim h (fun v61 h ↦ Exists.elim h (fun v62 h ↦ Exists.elim h (fun v63 h ↦ ?_))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
    a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38,
    a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, rfl⟩ := h
  obtain ⟨hloop, hsw⟩ := loop_of hchild (L := L) (by omega) a0 a1 a2 a3 a4
  obtain ⟨_, rfl⟩ := familyGet_lit L a5
  obtain ⟨_, rfl⟩ := familyGet_lit L a10
  obtain ⟨rfl, rfl, rfl, rfl⟩ := lutA_of_samples a6 a11 a7 a12
  have hs := lut_branches_gen hn hpow (L := L) hL hq hq1 hq' layout hlayout
    hexact (sig tab W) hodd hsm hg' hg hinput hloop hsw (hhelp ⟨L, by omega⟩)
    a16 a17 a18 a21 a22 a23 a19 a20 a24 a25 a26
    (Xb := ![v36, v47]) (Fin.forall_fin_two.2 ⟨a27, a38⟩)
    (D' := ![v37, v48]) (Fin.forall_fin_two.2 ⟨a28, a39⟩)
    (D := ![v38, v49]) (Fin.forall_fin_two.2 ⟨a29, a40⟩)
    (M := ![![v39, v50], ![v41, v52]]) (Fin.forall_fin_two.2 ⟨a30, a41⟩)
    (Fin.forall_fin_two.2 ⟨a32, a43⟩) (Asw := ![![v40, v51], ![v42, v53]])
    (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨a31, a42⟩, Fin.forall_fin_two.2 ⟨a33, a44⟩⟩)
    (Z := ![v43, v54]) (Fin.forall_fin_two.2 ⟨a34, a45⟩)
    (E' := ![v44, v55]) (Fin.forall_fin_two.2 ⟨a35, a46⟩)
    (E := ![v45, v56]) (Fin.forall_fin_two.2 ⟨a36, a47⟩)
    a8 a9 a13 a14 a15
    (Y := ![v60, v62]) (Fin.forall_fin_two.2 ⟨a51, a53⟩)
    (X := ![v61, v63]) (Fin.forall_fin_two.2 ⟨a52, a54⟩)
    (K := ![![v46, v57], ![v58, v59]]) (Fin.forall_fin_two.2 ⟨a37, a48⟩)
    (Fin.forall_fin_two.2 ⟨a49, a50⟩)
  exact encRel_of (hs.add hoff)
    (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨rfl, rfl⟩, Fin.forall_fin_two.2 ⟨rfl, rfl⟩⟩)
    (Fin.forall_fin_two.2 ⟨rfl, rfl⟩)

/-- The exported smudge weight `c · lift(P^(i mod d))` is `matrixPolynomial [c P^(i mod d)]`. -/
theorem smudge_weight_eq {n q : Nat} (c P : Int) (i d : Nat) :
    (matrixMulScalarLeft (matrixPolynomial [c])
        (liftInteger (P ^ (Int.fmod (Int.ofNat i) (d : Int)).toNat)) : ExactMatrix q n 1 1) =
      matrixPolynomial [c * P ^ (i % d)] := by
  rw [fmod_ofNat]
  have ht : ((i : Int) % (d : Int)).toNat = i % d := by omega
  funext r k
  simp [matrixMulScalarLeft, matrixPolynomial, liftInteger, ht]

set_option maxHeartbeats 64000000 in
/-- **A smudge member** (`smudge.member`, a width-`W` readout LUT finalized with a weight): member
`i` is the readout's `Δ' ν` (`LutMemberReadout.rel`'s hypotheses) times the weight
`c P^(i mod d)` (the exported `matrixPolynomial [c] · lift(P^(i mod D))`), placed at
`X^(⌊i/d⌋ mod N)` (`smudge_member_weight_gen`); `hD`, `hNN` fix the exported period and modulus
of the placement. -/
theorem SmudgeMember.rel {n q q' m m' W : Nat} {backend : BackendContext}
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
    (hN : 0 < N) (hN2 : N ≤ 2 * n) (L : Nat) (hℓ : ℓ = (L : Int)) (hL : L + 1 = W)
    (hn : 1 < n) {log : Nat} (hpow : n = 2 ^ log)
    {p : Nat} (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p)
    (layout : RegularLayout q') (hlayout : backend.regularLayout q' n = some layout)
    (hexact : layout.droppedModuli = 0) (hodd : ∀ k, Odd (sig tab W k).toNat)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix q' n 1 m} {g : ExactMatrix q n 1 m}
    (hg' : gadgetMatrixRuns backend base digits g') (hg : centeredRebaseRuns g' g) {B : Nat}
    (hinput : Rel sm (inA x0 x1 x2 x3 x4 x5)
      (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) (inC x0 x1 x2 x3 x4 x5) B)
    {Bh : Nat}
    (hhelp : ∀ k : Fin W, Rel sm (lutAG hashModel key q m tag level (sig tab W k))
      (fun j ↦ reducePoly q n ((p : Int) *
        unitPair (substitution (sig tab W k).toNat (hodd k) sm) j) • g) ![hf0 k, hf1 k] Bh)
    (hchild : LutStepHyp child hf0 key hf1 x0 x1 x2 x3 x4 x5 f00 f01 f10 f11 sm
      (lutBranchTargetG (q := q) (p := p) (sig tab W) hodd μ sp (fam4 f00 f01 f10 f11))
      (lutBranchBoundG n m bound p B Bh))
    {Δ' ν : ErrorPoly n} {gq : ExactMatrix q' n 1 m'} {C : Fin W → ExactPoly q' n} {BB : Nat}
    (hswitch : ∀ k : Fin W, ∃ e : Fin 2 → ErrorMatrix n 1 m', (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly q' n (substitution (sig tab W k).toNat (hodd k) sp) •
          fam4 f00 f01 f10 f11 0 j k + fam4 f00 f01 f10 f11 1 j k =
        (C k * reducePoly q' n (Δ' * unitPair sp j)) • gq + reduceMatrix q' n 1 m' (e j))
    (hμ : ∀ k, polyNorm (substitution (sig tab W k).toNat (hodd k) μ) ≤ 1)
    (hwalsh : ∑ k, reducePoly q' n (substitution (sig tab W k).toNat (hodd k) μ) * C k =
      reducePoly q' n ν)
    (layoutG : RegularLayout q') (hlayoutG : backend.regularLayout q' n = some layoutG)
    (hexactG : layoutG.droppedModuli = 0) (hG : gadgetMatrixRuns backend baseG digitsG gq) :
    EncRel sm outputs (fun j ↦ reducePoly q' n (Xe (n := n) (i / d % N)) •
        reducePoly q' n ((((c * P ^ (i % d) : Int)) : ErrorPoly n) * (Δ' * ν) * unitPair sp j) •
          (1 : ExactMatrix q' n 1 1))
      (m' * ((L * lutBranchBoundG n m bound p B Bh + lutBranchBoundG n m bound p B Bh +
        (n + 2) * (p / 2)) / p + W * (n * 1 * BB)) * (baseG.toNat / 2)) := by
  subst hmw hℓ hD hNN
  refine Exists.elim h (fun v0 h ↦ Exists.elim h (fun v1 h ↦ Exists.elim h (fun v2 h ↦ Exists.elim h (fun v3 h ↦ Exists.elim h (fun v4 h ↦ Exists.elim h (fun v5 h ↦ Exists.elim h (fun v6 h ↦ Exists.elim h (fun v7 h ↦ Exists.elim h (fun v8 h ↦ Exists.elim h (fun v9 h ↦ Exists.elim h (fun v10 h ↦ Exists.elim h (fun v11 h ↦ Exists.elim h (fun v12 h ↦ Exists.elim h (fun v13 h ↦ Exists.elim h (fun v14 h ↦ Exists.elim h (fun v15 h ↦ Exists.elim h (fun v16 h ↦ Exists.elim h (fun v17 h ↦ Exists.elim h (fun v18 h ↦ Exists.elim h (fun v19 h ↦ Exists.elim h (fun v20 h ↦ Exists.elim h (fun v21 h ↦ Exists.elim h (fun v22 h ↦ Exists.elim h (fun v23 h ↦ Exists.elim h (fun v24 h ↦ Exists.elim h (fun v25 h ↦ Exists.elim h (fun v26 h ↦ Exists.elim h (fun v27 h ↦ Exists.elim h (fun v28 h ↦ Exists.elim h (fun v29 h ↦ Exists.elim h (fun v30 h ↦ Exists.elim h (fun v31 h ↦ Exists.elim h (fun v32 h ↦ Exists.elim h (fun v33 h ↦ Exists.elim h (fun v34 h ↦ Exists.elim h (fun v35 h ↦ Exists.elim h (fun v36 h ↦ Exists.elim h (fun v37 h ↦ Exists.elim h (fun v38 h ↦ Exists.elim h (fun v39 h ↦ Exists.elim h (fun v40 h ↦ Exists.elim h (fun v41 h ↦ Exists.elim h (fun v42 h ↦ Exists.elim h (fun v43 h ↦ Exists.elim h (fun v44 h ↦ Exists.elim h (fun v45 h ↦ Exists.elim h (fun v46 h ↦ Exists.elim h (fun v47 h ↦ Exists.elim h (fun v48 h ↦ Exists.elim h (fun v49 h ↦ Exists.elim h (fun v50 h ↦ Exists.elim h (fun v51 h ↦ Exists.elim h (fun v52 h ↦ Exists.elim h (fun v53 h ↦ Exists.elim h (fun v54 h ↦ Exists.elim h (fun v55 h ↦ Exists.elim h (fun v56 h ↦ Exists.elim h (fun v57 h ↦ Exists.elim h (fun v58 h ↦ Exists.elim h (fun v59 h ↦ Exists.elim h (fun v60 h ↦ Exists.elim h (fun v61 h ↦ Exists.elim h (fun v62 h ↦ Exists.elim h (fun v63 h ↦ Exists.elim h (fun v64 h ↦ Exists.elim h (fun v65 h ↦ Exists.elim h (fun v66 h ↦ Exists.elim h (fun v67 h ↦ Exists.elim h (fun v68 h ↦ Exists.elim h (fun v69 h ↦ Exists.elim h (fun v70 h ↦ Exists.elim h (fun v71 h ↦ ?_))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))
  obtain ⟨a0, a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11, a12, a13, a14, a15, a16, a17, a18, a19,
    a20, a21, a22, a23, a24, a25, a26, a27, a28, a29, a30, a31, a32, a33, a34, a35, a36, a37, a38,
    a39, a40, a41, a42, a43, a44, a45, a46, a47, a48, a49, a50, a51, a52, a53, a54, a55, a56, a57,
    a58, a59, a60, a61, a62, hout⟩ := h
  obtain ⟨hloop, hsw⟩ := loop_of hchild (L := L) (by omega) a0 a1 a2 a3 a4
  obtain ⟨_, rfl⟩ := familyGet_lit L a5
  obtain ⟨_, rfl⟩ := familyGet_lit L a10
  obtain ⟨rfl, rfl, rfl, rfl⟩ := lutA_of_samples a6 a11 a7 a12
  have hR := lut_round_gen (by omega) hpow (L := L) hL hq hq1 hq' hp layout hlayout
    hexact (sig tab W) hodd hsm hg' hg hinput hloop hsw (hhelp ⟨L, by omega⟩)
    a16 a17 a18 a21 a22 a23 a19 a20 a24 a25 a26
    (Xb := ![v36, v50]) (Fin.forall_fin_two.2 ⟨a27, a41⟩)
    (D' := ![v37, v51]) (Fin.forall_fin_two.2 ⟨a28, a42⟩)
    (D := ![v38, v52]) (Fin.forall_fin_two.2 ⟨a29, a43⟩)
    (M := ![![v39, v53], ![v41, v55]]) (Fin.forall_fin_two.2 ⟨a30, a44⟩)
    (Fin.forall_fin_two.2 ⟨a32, a46⟩) (Asw := ![![v40, v54], ![v42, v56]])
    (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨a31, a45⟩, Fin.forall_fin_two.2 ⟨a33, a47⟩⟩)
    (Z := ![v43, v57]) (Fin.forall_fin_two.2 ⟨a34, a48⟩)
    (E' := ![v44, v58]) (Fin.forall_fin_two.2 ⟨a35, a49⟩)
    (E := ![v45, v59]) (Fin.forall_fin_two.2 ⟨a36, a50⟩)
    a8 a9 a13 a14 a15
    (Y := ![v66, v69]) (Fin.forall_fin_two.2 ⟨a57, a60⟩)
    (X := ![v67, v70]) (Fin.forall_fin_two.2 ⟨a58, a61⟩)
    (K := ![![v46, v60], ![v62, v64]]) (Fin.forall_fin_two.2 ⟨a37, a51⟩)
    (Fin.forall_fin_two.2 ⟨a53, a55⟩) hswitch hμ hwalsh
    (K' := ![![v47, v61], ![v63, v65]])
    (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨a38, a52⟩, Fin.forall_fin_two.2 ⟨a54, a56⟩⟩)
    (R' := ![v68, v71]) (Fin.forall_fin_two.2 ⟨a59, a62⟩)
  rw [smudge_weight_eq] at a39
  rw [fdiv_fmod_ofNat] at hout
  subst hout
  exact encRel_of (smudge_member_weight_gen hn hq' layoutG hlayoutG hexactG hG hR
    (c * P ^ (i % d)) rfl a39 a40 (e := i / d % N) ((Nat.mod_lt _ hN).trans_le hN2))
    (Fin.forall_fin_two.2 ⟨Fin.forall_fin_two.2 ⟨rfl, rfl⟩, Fin.forall_fin_two.2 ⟨rfl, rfl⟩⟩)
    (Fin.forall_fin_two.2 ⟨rfl, rfl⟩)

end RefreshTopProof.GenIface
