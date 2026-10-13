import GenRel
import GenLut
import GenLoop

/-!
# LUT rounds, samplers and smudges on the encoding relation, for any parameter set

The encoding-level counterpart of `GenLut` and `GenSmudge`. A LUT of width `W` has branches
`σ_i`, `i < W`; the generated loop accumulates the first `L = W - 1` of them and carries the
switching ciphertext of the next one, and the last branch and the move down follow the loop.
Nothing names the ring dimension, the width, the stride, a modulus, a gadget width or the
decomposition bound.

* `LoopStateG`, `branchTargetG`, `lutBranchTargetG`, `lutBranchBoundG`: the loop state, the
  branch targets and the bound one branch adds.
* `lut_loop_gen`: any generated loop whose body adds branch `i` accumulates the first `count`.
* `lut_branches_gen`: the loop and the last branch.
* `lut_round_gen`: a LUT round with any interpolation (`hwalsh`); `lut_round_value_gen`: the
  full-period round on `X^(s z)` (`n = s 2^lw`) with the LUT coefficients of the table `T`,
  an encoding of `Δ' (-1)^⌊z / 2^lw⌋ T(z mod 2^lw)`.
* `lut_finish_sum_gen`: `lut_finish` over any finite family of branches.
* `sampler_carrier_gen`: a PRF sampler (the mask or a fresh-noise column): every member's
  sampling LUT plus the binary offset, placed at `X^(pl o)`, summed and moved down once.
* `smudgeValueG`, `smudge_member_gen`, `smudge_chain_gen`: the smudge of `C` columns of `d`
  digits: member `o` finalized with the digit weight `w (o mod d)` and placed at `X^⌊o/d⌋`.
-/

namespace RefreshTopProof.Gen

open Mxx.Primitives MxxRuntime MxxExponentLut
open scoped BigOperators

variable {n : Nat}

/-! ## The loop state and the branches -/

/-- The six accumulated sums and the carried switching ciphertext of a LUT loop state. -/
abbrev LoopStateG (n q q' m : Nat) :=
  ExactMatrix q n 1 m × ExactMatrix q n 1 m × ExactMatrix q n 1 m × ExactMatrix q n 1 m ×
    ExactMatrix q n 1 m × ExactMatrix q n 1 m × ExactMatrix q' n 1 m × ExactMatrix q' n 1 m ×
      ExactMatrix q' n 1 m × ExactMatrix q' n 1 m × Unit

/-- The accumulated row. -/
def LoopStateG.row {n q q' m : Nat} (s : LoopStateG n q q' m) : Fin 2 → ExactMatrix q n 1 m :=
  ![s.1, s.2.1]

/-- The accumulated key. -/
def LoopStateG.key {n q q' m : Nat} (s : LoopStateG n q q' m) :
    Fin 2 → Fin 2 → ExactMatrix q n 1 m :=
  ![![s.2.2.1, s.2.2.2.1], ![s.2.2.2.2.1, s.2.2.2.2.2.1]]

/-- The carried switching ciphertext. -/
def LoopStateG.switch {n q q' m : Nat} (s : LoopStateG n q q' m) :
    Fin 2 → Fin 2 → ExactMatrix q' n 1 m :=
  ![![s.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.1], ![s.2.2.2.2.2.2.2.2.1, s.2.2.2.2.2.2.2.2.2.1]]

/-- The target of branch `i` of a LUT with automorphisms `sigma` over the switching families. -/
noncomputable def branchTargetG {q q' p m W : Nat} (sigma : Fin W → Int)
    (hodd : ∀ i, Odd (sigma i).toNat) (μ sp : ErrorPoly n)
    (fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q' n 1 m) (i : Fin W) :
    Fin 2 → ExactMatrix q n 1 m :=
  fun j ↦ scaleUpMatrix q p (reducePoly q' n (substitution (sigma i).toNat (hodd i) μ) •
    (reducePoly q' n (substitution (sigma i).toNat (hodd i) sp) • fam 0 j i + fam 1 j i))

/-- Branch `i`, zero past the `W` branches. -/
noncomputable def lutBranchTargetG {q q' p m W : Nat} (sigma : Fin W → Int)
    (hodd : ∀ i, Odd (sigma i).toNat) (μ sp : ErrorPoly n)
    (fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q' n 1 m) (i : Nat) : Fin 2 → ExactMatrix q n 1 m :=
  if h : i < W then branchTargetG (q := q) (p := p) sigma hodd μ sp fam ⟨i, h⟩ else 0

/-- The bound one branch adds: the helper (bound `Bh`) and the routed input (bound `B`) act
through decompositions of bound `bound`, and the switched key is moved down by `p`. -/
abbrev lutBranchBoundG (n m bound p B Bh : Nat) : Nat :=
  (m + m) * n * Bh * bound + (m + m) * n * B * bound + (n * 1 * (p / 2) + p / 2)

/-! ## The loop -/

/-- **The LUT loop.** A generated loop (any state type) whose iteration `i` adds `T i` to the
accumulated relation, with bound `Bb`, and moves the carried switching ciphertext from branch
`i` to branch `i + 1`, accumulates the first `count < W` branches. -/
theorem lut_loop_gen {S : Type} {q q' m W : Nat} {sm : ErrorPoly n}
    (key : S → Fin 2 → Fin 2 → ExactMatrix q n 1 m) (row : S → Fin 2 → ExactMatrix q n 1 m)
    (switch : S → Fin 2 → Fin 2 → ExactMatrix q' n 1 m) (T : Nat → Fin 2 → ExactMatrix q n 1 m)
    (fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q' n 1 m) (Bb : Nat)
    {body : Nat → S → S → Prop} {count : Nat} (hcount : count < W) {init final : S}
    (hinit : Rel sm (key init) 0 (row init) 0)
    (hinitSwitch : switch init = fun k j ↦ fam k j ⟨0, by omega⟩)
    (hstep : ∀ i (hi : i + 1 < W) cur nxt (Tc : Fin 2 → ExactMatrix q n 1 m) (Bs : Nat),
      body i cur nxt → switch cur = (fun k j ↦ fam k j ⟨i, by omega⟩) →
      Rel sm (key cur) Tc (row cur) Bs →
      Rel sm (key nxt) (Tc + T i) (row nxt) (Bs + Bb) ∧
        switch nxt = fun k j ↦ fam k j ⟨i + 1, hi⟩)
    (hruns : MxxIR.IterRuns body count init final) :
    Rel sm (key final) (∑ i ∈ Finset.range count, T i) (row final) (count * Bb) ∧
      switch final = fun k j ↦ fam k j ⟨count, hcount⟩ := by
  refine MxxIR.IterRuns.invariant
    (Invariant := fun k (s : S) ↦ ∀ hk : k < W,
      Rel sm (key s) (∑ i ∈ Finset.range k, T i) (row s) (k * Bb) ∧
        switch s = fun k' j ↦ fam k' j ⟨k, hk⟩) ?_ ?_ hruns hcount
  · intro hk
    refine ⟨?_, hinitSwitch⟩
    rw [Finset.range_zero, Finset.sum_empty, zero_mul]
    exact hinit
  · intro i current next hinv hbody hk
    obtain ⟨hrel, hsw⟩ := hinv (by omega)
    obtain ⟨hrel', hsw'⟩ := hstep i hk current next _ _ hbody hsw hrel
    refine ⟨?_, hsw'⟩
    rw [Finset.sum_range_succ, add_mul, one_mul]
    exact hrel'

/-! ## The last branch and the round -/

set_option maxHeartbeats 64000000 in
/-- The `W = L + 1` branches of a LUT before its move down: the loop's `L` (`hloop`) and the
last. -/
theorem lut_branches_gen (hn : 0 < n) {log : Nat} (hpow : n = 2 ^ log)
    {q q' p m m' bound W L : Nat} (hL : L + 1 = W) (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q')
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    (sigma : Fin W → Int) (hodd : ∀ i, Odd (sigma i).toNat)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix q' n 1 m} {g : ExactMatrix q n 1 m}
    (hg' : gadgetMatrixRuns backend base digits g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix q n 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (hinput : Rel sm A (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) c B)
    {fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q' n 1 m'} {final : LoopStateG n q q' m'} {Bl : Nat}
    (hloop : Rel sm final.key (∑ i ∈ Finset.range L, lutBranchTargetG (q := q) (p := p)
      sigma hodd μ sp fam i) final.row Bl)
    (hloopSwitch : final.switch = fun k j ↦ fam k j ⟨L, by omega⟩)
    {h3 : Fin 2 → ExactMatrix q n 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {Bh : Nat}
    (hhelp : Rel sm Ad (fun j ↦ reducePoly q n ((p : Int) *
        unitPair (substitution (sigma ⟨L, by omega⟩).toNat (hodd ⟨L, by omega⟩) sm) j) • g) h3 Bh)
    {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q n 1 m}
    (hr0 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (c 0) rc0)
    (hr1 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (A 0 0) rA00)
    (hr2 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (A 1 0) rA10)
    (hr3 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (c 1) rc1)
    (hr4 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (A 0 1) rA01)
    (hr5 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (A 1 1) rA11)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows rc0 rA00 l01) (hl : concatRows l01 rA10 l)
    (hr01 : concatRows rc1 rA01 r01) (hr : concatRows r01 rA11 r) (hboth : concatColumns l r both)
    {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m'}
    (hXb : ∀ j, concatRows (final.switch 0 j) (final.switch 1 j) (Xb j))
    {D' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {basis : List Nat} {Asw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    (hAsw : ∀ k j, blockModSwitchRuns basis 1 (M k j) (Asw k j))
    {Z : Fin 2 → ExactMatrix q' n (1 + 1) m'} (hZ : ∀ j, concatRows (Asw 0 j) (Asw 1 j) (Z j))
    {E' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hE' : ∀ j, gadgetDecomposeRuns backend base digits (Z j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m + m) m'} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {hl01' hr01' : ExactMatrix q n (1 + 1) m} {hl' hr' : ExactMatrix q n (1 + 1 + 1) m}
    {H : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hhl01 : concatRows (h3 0) (Ad 0 0) hl01') (hhl : concatRows hl01' (Ad 1 0) hl')
    (hhr01 : concatRows (h3 1) (Ad 0 1) hr01') (hhr : concatRows hr01' (Ad 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {Y X : Fin 2 → ExactMatrix q n 1 m'} {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul H (E j)) 0 1 0 m' (Y j))
    (hX : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (X j))
    (hK0 : ∀ j, sliceMatrix (matrixMul H (E j)) 1 2 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixMul H (E j)) 2 3 0 m' (K 1 j)) :
    Rel sm (fun k j ↦ final.key k j + K k j)
      (fun j ↦ ∑ i : Fin W, branchTargetG (q := q) (p := p) sigma hodd μ sp fam i j)
      (fun j ↦ final.row j + (Y j + X j)) (Bl + lutBranchBoundG n m bound p B Bh) := by
  have hbranch := lut_branch (q := q) (q' := q') (p := p) (m := m) (m' := m') (log := log)
    (bound := bound) (index := sigma ⟨L, by omega⟩) hn hpow hq hq1 hq' layout hlayout hexact hg'
    hg hsm hinput hr0 hr1 hr2 hr3 hr4 hr5 hl01 hl hr01 hr hboth (bsw := final.switch) hXb hD' hD
    hM0 hM1 hAsw hZ hE' hE (h := h3) (hodd _) hhelp hhl01 hhl hhr01 hhr hH hY hX hK0 hK1
  rw [hloopSwitch] at hbranch
  have hsum := hloop.add hbranch
  refine hsum.target_eq ?_
  funext j
  subst hL
  rw [Fin.sum_univ_castSucc, Pi.add_apply, Finset.sum_apply, Finset.sum_range]
  refine congrArg₂ (· + ·) (Finset.sum_congr rfl fun i _ ↦ ?_) ?_
  · rw [lutBranchTargetG, dif_pos (by omega)]
    rfl
  · rfl

set_option maxHeartbeats 64000000 in
/-- **A LUT round**: the loop's `L` branches (`hloop`), the last branch and the move down to
`q'`, for any interpolation: if the switching ciphertexts encode the coefficients `C_i` and
`Σ_i σ_i(μ) C_i = ν`, the result encodes `Δ' ν` at `q'` against `gq`. -/
theorem lut_round_gen (hn : 0 < n) {log : Nat} (hpow : n = 2 ^ log)
    {q q' p m m' bound W L : Nat} (hL : L + 1 = W) (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q')
    (hp : 0 < p)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    (sigma : Fin W → Int) (hodd : ∀ i, Odd (sigma i).toNat)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix q' n 1 m} {g : ExactMatrix q n 1 m}
    (hg' : gadgetMatrixRuns backend base digits g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix q n 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (hinput : Rel sm A (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) c B)
    {fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q' n 1 m'} {final : LoopStateG n q q' m'} {Bl : Nat}
    (hloop : Rel sm final.key (∑ i ∈ Finset.range L, lutBranchTargetG (q := q) (p := p)
      sigma hodd μ sp fam i) final.row Bl)
    (hloopSwitch : final.switch = fun k j ↦ fam k j ⟨L, by omega⟩)
    {h3 : Fin 2 → ExactMatrix q n 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {Bh : Nat}
    (hhelp : Rel sm Ad (fun j ↦ reducePoly q n ((p : Int) *
        unitPair (substitution (sigma ⟨L, by omega⟩).toNat (hodd ⟨L, by omega⟩) sm) j) • g) h3 Bh)
    {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q n 1 m}
    (hr0 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (c 0) rc0)
    (hr1 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (A 0 0) rA00)
    (hr2 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (A 1 0) rA10)
    (hr3 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (c 1) rc1)
    (hr4 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (A 0 1) rA01)
    (hr5 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (A 1 1) rA11)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows rc0 rA00 l01) (hl : concatRows l01 rA10 l)
    (hr01 : concatRows rc1 rA01 r01) (hr : concatRows r01 rA11 r) (hboth : concatColumns l r both)
    {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m'}
    (hXb : ∀ j, concatRows (final.switch 0 j) (final.switch 1 j) (Xb j))
    {D' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {basis : List Nat} {Asw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    (hAsw : ∀ k j, blockModSwitchRuns basis 1 (M k j) (Asw k j))
    {Z : Fin 2 → ExactMatrix q' n (1 + 1) m'} (hZ : ∀ j, concatRows (Asw 0 j) (Asw 1 j) (Z j))
    {E' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hE' : ∀ j, gadgetDecomposeRuns backend base digits (Z j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m + m) m'} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {hl01' hr01' : ExactMatrix q n (1 + 1) m} {hl' hr' : ExactMatrix q n (1 + 1 + 1) m}
    {H : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hhl01 : concatRows (h3 0) (Ad 0 0) hl01') (hhl : concatRows hl01' (Ad 1 0) hl')
    (hhr01 : concatRows (h3 1) (Ad 0 1) hr01') (hhr : concatRows hr01' (Ad 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {Y X : Fin 2 → ExactMatrix q n 1 m'} {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul H (E j)) 0 1 0 m' (Y j))
    (hX : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (X j))
    (hK0 : ∀ j, sliceMatrix (matrixMul H (E j)) 1 2 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixMul H (E j)) 2 3 0 m' (K 1 j))
    {Δ' ν : ErrorPoly n} {gq : ExactMatrix q' n 1 m'} {C : Fin W → ExactPoly q' n} {BB : Nat}
    (hswitch : ∀ i : Fin W, ∃ e : Fin 2 → ErrorMatrix n 1 m', (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly q' n (substitution (sigma i).toNat (hodd i) sp) • fam 0 j i + fam 1 j i =
        (C i * reducePoly q' n (Δ' * unitPair sp j)) • gq + reduceMatrix q' n 1 m' (e j))
    (hμ : ∀ i, polyNorm (substitution (sigma i).toNat (hodd i) μ) ≤ 1)
    (hwalsh : ∑ i, reducePoly q' n (substitution (sigma i).toNat (hodd i) μ) * C i =
      reducePoly q' n ν)
    {basis' : List Nat} {K' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    {R' : Fin 2 → ExactMatrix q' n 1 m'}
    (hK' : ∀ k j, blockModSwitchRuns basis' 1 (final.key k j + K k j) (K' k j))
    (hR' : ∀ j, blockModSwitchRuns basis' 1 (final.row j + (Y j + X j)) (R' j)) :
    Rel sm K' (fun j ↦ reducePoly q' n (Δ' * ν * unitPair sp j) • gq) R'
      ((Bl + lutBranchBoundG n m bound p B Bh + (n + 2) * (p / 2)) / p + W * (n * 1 * BB)) := by
  have hsum' := lut_branches_gen hn hpow hL hq hq1 hq' layout hlayout hexact sigma hodd hsm hg' hg
    hinput hloop hloopSwitch hhelp hr0 hr1 hr2 hr3 hr4 hr5 hl01 hl hr01 hr hboth hXb hD' hD hM0 hM1
    hAsw hZ hE' hE hhl01 hhl hhr01 hhr hH hY hX hK0 hK1
  choose eS heS hS using hswitch
  exact lut_finish (W := W) (q := q) (q' := q') (p := p) (m' := m') hn hq hq1 hq' hp
    (sp := sp) (μ := μ) (Δ' := Δ') hsm (fun i ↦ substitution (sigma i).toNat (hodd i)) hμ
    (bsw := fun i k j ↦ fam k j i) hsum' (g' := gq) (C := C)
    (fun i j ↦ hS i j) (fun i j ↦ heS i j) hwalsh hK' hR'

theorem polyNorm_substitution_Xe_gen (hn : 1 < n) {index : Nat} (hodd : Odd index) (k : Nat) :
    polyNorm (substitution index hodd (Xe (n := n) k)) ≤ 1 := by
  rw [substitution_root_pow]
  exact polyNorm_Xe hn _

set_option maxHeartbeats 64000000 in
/-- **The full-period LUT round** of width `2^lw` on the state `X^(s z)`, `n = s 2^lw`, with the
LUT coefficients of the table `T` (the automorphisms `σ_i ≡ 1 + 2 π(i) (mod 2^(lw+1))` for a
permutation `π`): an encoding of `Δ' (-1)^⌊z / 2^lw⌋ T(z mod 2^lw)` at `q'`. -/
theorem lut_round_value_gen (hn : 1 < n) {log : Nat} (hpow : n = 2 ^ log)
    {q q' p m m' bound lw s L : Nat} (hL : L + 1 = 2 ^ lw) (hns : n = s * 2 ^ lw)
    (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    (sigma : Fin (2 ^ lw) → Int) (hodd : ∀ i, Odd (sigma i).toNat)
    (perm : Fin (2 ^ lw) ≃ Fin (2 ^ lw))
    (hindex : ∀ i, (sigma i).toNat % (2 * 2 ^ lw) = 1 + 2 * 1 * (perm i).val)
    {sm sp : ErrorPoly n} (hsm : polyNorm sm ≤ 1) {z : Nat}
    {g' : ExactMatrix q' n 1 m} {g : ExactMatrix q n 1 m}
    (hg' : gadgetMatrixRuns backend base digits g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix q n 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (hinput : Rel sm A (fun j ↦ reducePoly q n ((p : Int) * Xe (n := n) (s * z) *
      unitPair sp j) • g) c B)
    {fam : Fin 2 → Fin 2 → Fin (2 ^ lw) → ExactMatrix q' n 1 m'} {final : LoopStateG n q q' m'}
    {Bl : Nat}
    (hloop : Rel sm final.key (∑ i ∈ Finset.range L, lutBranchTargetG (q := q) (p := p)
      sigma hodd (Xe (s * z)) sp fam i) final.row Bl)
    (hloopSwitch : final.switch = fun k j ↦ fam k j ⟨L, by omega⟩)
    {h3 : Fin 2 → ExactMatrix q n 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {Bh : Nat}
    (hhelp : Rel sm Ad (fun j ↦ reducePoly q n ((p : Int) *
        unitPair (substitution (sigma ⟨L, by omega⟩).toNat (hodd ⟨L, by omega⟩) sm) j) • g) h3 Bh)
    {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q n 1 m}
    (hr0 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (c 0) rc0)
    (hr1 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (A 0 0) rA00)
    (hr2 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (A 1 0) rA10)
    (hr3 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (c 1) rc1)
    (hr4 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (A 0 1) rA01)
    (hr5 : ringAutomorphismRuns (sigma ⟨L, by omega⟩) (A 1 1) rA11)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows rc0 rA00 l01) (hl : concatRows l01 rA10 l)
    (hr01 : concatRows rc1 rA01 r01) (hr : concatRows r01 rA11 r) (hboth : concatColumns l r both)
    {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m'}
    (hXb : ∀ j, concatRows (final.switch 0 j) (final.switch 1 j) (Xb j))
    {D' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {basis : List Nat} {Asw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    (hAsw : ∀ k j, blockModSwitchRuns basis 1 (M k j) (Asw k j))
    {Z : Fin 2 → ExactMatrix q' n (1 + 1) m'} (hZ : ∀ j, concatRows (Asw 0 j) (Asw 1 j) (Z j))
    {E' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hE' : ∀ j, gadgetDecomposeRuns backend base digits (Z j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m + m) m'} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {hl01' hr01' : ExactMatrix q n (1 + 1) m} {hl' hr' : ExactMatrix q n (1 + 1 + 1) m}
    {H : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hhl01 : concatRows (h3 0) (Ad 0 0) hl01') (hhl : concatRows hl01' (Ad 1 0) hl')
    (hhr01 : concatRows (h3 1) (Ad 0 1) hr01') (hhr : concatRows hr01' (Ad 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {Y Xr : Fin 2 → ExactMatrix q n 1 m'} {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul H (E j)) 0 1 0 m' (Y j))
    (hX : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (Xr j))
    (hK0 : ∀ j, sliceMatrix (matrixMul H (E j)) 1 2 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixMul H (E j)) 2 3 0 m' (K 1 j))
    {T : Fin (2 ^ lw) → ErrorPoly n} {Δ' : ErrorPoly n} {gq : ExactMatrix q' n 1 m'}
    {inverse : Nat}
    (hinverse : (inverse : ExactPoly q' n) * ((2 ^ lw : Nat) : ExactPoly q' n) = 1) {BB : Nat}
    (hswitch : ∀ i : Fin (2 ^ lw), ∃ e : Fin 2 → ErrorMatrix n 1 m', (∀ j, CoeffBound (e j) BB) ∧
      ∀ j, reducePoly q' n (substitution (sigma i).toNat (hodd i) sp) • fam 0 j i + fam 1 j i =
        (lutFourierCoefficient (2 ^ lw) (X q' ^ s) (inverse : ExactPoly q' n)
            (fun k ↦ reducePoly q' n (T k)) (sigma i).toNat *
          reducePoly q' n (Δ' * unitPair sp j)) • gq + reduceMatrix q' n 1 m' (e j))
    {basis' : List Nat} {K' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    {R' : Fin 2 → ExactMatrix q' n 1 m'}
    (hK' : ∀ k j, blockModSwitchRuns basis' 1 (final.key k j + K k j) (K' k j))
    (hR' : ∀ j, blockModSwitchRuns basis' 1 (final.row j + (Y j + Xr j)) (R' j)) :
    Rel sm K' (fun j ↦ reducePoly q' n (Δ' * ((((-1 : Int) ^ (z / 2 ^ lw) : Int) : ErrorPoly n) *
        T ⟨z % 2 ^ lw, Nat.mod_lt _ (by positivity)⟩) * unitPair sp j) • gq) R'
      ((Bl + lutBranchBoundG n m bound p B Bh + (n + 2) * (p / 2)) / p +
        2 ^ lw * (n * 1 * BB)) :=
  lut_round_gen (by omega) hpow hL hq hq1 hq' hp layout hlayout hexact sigma hodd hsm hg' hg hinput
    hloop hloopSwitch hhelp hr0 hr1 hr2 hr3 hr4 hr5 hl01 hl hr01 hr hboth hXb hD' hD hM0 hM1 hAsw
    hZ hE' hE hhl01 hhl hhr01 hhr hH hY hX hK0 hK1 hswitch
    (fun i ↦ polyNorm_substitution_Xe_gen hn (hodd i) _)
    (lut_gen_value (q := q') (inverse := inverse) (s := s) (log := lw) hns (by omega) T hinverse
      (fun i ↦ (sigma i).toNat) hodd perm hindex z) hK' hR'

/-! ## Moving a sum of branches down -/

theorem smul_scaleUpMatrix_gen (hn : 0 < n) {q q' p rows columns : Nat} (hq : q = p * q')
    (hq' : 1 < q') (a : ErrorPoly n) (x : ExactMatrix q' n rows columns) :
    reducePoly q n a • scaleUpMatrix q p x = scaleUpMatrix q p (reducePoly q' n a • x) := by
  funext r c
  simp only [scaleUpMatrix, Matrix.smul_apply, smul_eq_mul, scaleUp_reducePoly_mul hq hq' hn]

set_option maxHeartbeats 4000000 in
/-- `lut_finish` over any finite family `ι` of branches with multipliers `a` and secret
automorphisms `b`. -/
theorem lut_finish_sum_gen (hn : 0 < n) {ι : Type} [Fintype ι] [DecidableEq ι] {q q' p m' : Nat}
    (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p) {sm sp ν Δ' : ErrorPoly n}
    (hsm : polyNorm sm ≤ 1) (a b : ι → ErrorPoly n) (ha : ∀ i, polyNorm (a i) ≤ 1)
    {bsw : ι → Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    {Ksum : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {Rsum : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hsum : Rel sm Ksum (fun j ↦ ∑ i, scaleUpMatrix q p (reducePoly q' n (a i) •
      (reducePoly q' n (b i) • bsw i 0 j + bsw i 1 j))) Rsum Bs)
    {g' : ExactMatrix q' n 1 m'} {C : ι → ExactPoly q' n} {eB : ι → Fin 2 → ErrorMatrix n 1 m'}
    {BB : Nat} (hswitch : ∀ i j, reducePoly q' n (b i) • bsw i 0 j + bsw i 1 j =
      (C i * reducePoly q' n (Δ' * unitPair sp j)) • g' + reduceMatrix q' n 1 m' (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (hwalsh : ∑ i, reducePoly q' n (a i) * C i = reducePoly q' n ν) {basis : List Nat}
    {K' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'} {R' : Fin 2 → ExactMatrix q' n 1 m'}
    (hK' : ∀ k j, blockModSwitchRuns basis 1 (Ksum k j) (K' k j))
    (hR' : ∀ j, blockModSwitchRuns basis 1 (Rsum j) (R' j)) :
    Rel sm K' (fun j ↦ reducePoly q' n (Δ' * ν * unitPair sp j) • g') R'
      ((Bs + (n + 2) * (p / 2)) / p + Fintype.card ι * (n * 1 * BB)) := by
  let f : Fin 2 → ErrorMatrix n 1 m' := fun j ↦ ∑ i, a i • eB i j
  have htarget : ∀ j, (∑ i, scaleUpMatrix q p (reducePoly q' n (a i) •
      (reducePoly q' n (b i) • bsw i 0 j + bsw i 1 j))) =
      scaleUpMatrix q p (reducePoly q' n (Δ' * ν * unitPair sp j) • g' +
        reduceMatrix q' n 1 m' (f j)) := by
    intro j
    simp only [hswitch]
    funext row column
    simp only [scaleUpMatrix, Matrix.sum_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul,
      reduceMatrix, f]
    rw [← scaleUp_sum hq hq' hn]
    congr 1
    have hterm : ∀ i ∈ (Finset.univ : Finset ι), reducePoly q' n (a i) *
        (C i * reducePoly q' n (Δ' * unitPair sp j) * g' row column +
          reducePoly q' n (eB i j row column)) =
        (reducePoly q' n (a i) * C i) * (reducePoly q' n (Δ' * unitPair sp j) * g' row column) +
          reducePoly q' n (a i * eB i j row column) := by
      intro i _
      simp only [map_mul]
      ring
    rw [Finset.sum_congr rfl hterm, Finset.sum_add_distrib, ← Finset.sum_mul, hwalsh,
      map_sum]
    simp only [map_mul, Matrix.sum_apply, Matrix.smul_apply, smul_eq_mul]
    ring
  have hsum' : Rel sm Ksum (fun j ↦ scaleUpMatrix q p
      (reducePoly q' n (Δ' * ν * unitPair sp j) • g' + reduceMatrix q' n 1 m' (f j))) Rsum Bs := by
    obtain ⟨e, he, hc⟩ := hsum
    exact ⟨e, he, fun j ↦ by rw [hc j]; simp only []; rw [htarget j]⟩
  have hdown := hsum'.modDown hn hq hq1 hq' hp hsm (fun j ↦ rfl)
    (fun k j ↦ blockModSwitchRuns_scaleUp' hn hq (hK' k j))
    (fun j ↦ blockModSwitchRuns_scaleUp' hn hq (hR' j))
  have hf : ∀ j, CoeffBound (f j) (Fintype.card ι * (n * 1 * BB)) := by
    intro j
    have := coeffBound_sum Finset.univ (fun i _ ↦ coeffBound_smul hn (ha i) (heB i j))
    simpa [Finset.sum_const, Finset.card_univ] using this
  exact hdown.absorb hf

/-! ## A PRF sampler -/

/-- The branch multiplier of member `o`, part `b` (the member's sample or the offset) and branch
`i`: the placement `X^(pl o)` times `σ_i` of the part's state `X^(s z)`. -/
noncomputable def samplerMul {M W : Nat} (sigma : Fin W → Int) (hodd : ∀ i, Odd (sigma i).toNat)
    (pl : Fin M → Nat) (s : Nat) (ζ : Fin M → Nat) (zoff : Nat) (x : Fin M × Fin 2 × Fin W) :
    ErrorPoly n :=
  Xe (n := n) (pl x.1) * substitution (sigma x.2.2).toNat (hodd x.2.2)
    (Xe (n := n) (s * ![ζ x.1, zoff] x.2.1))

theorem polyNorm_samplerMul (hn : 1 < n) {M W : Nat} (sigma : Fin W → Int)
    (hodd : ∀ i, Odd (sigma i).toNat) (pl : Fin M → Nat) (s : Nat) (ζ : Fin M → Nat) (zoff : Nat)
    (x : Fin M × Fin 2 × Fin W) : polyNorm (samplerMul (n := n) sigma hodd pl s ζ zoff x) ≤ 1 := by
  simp only [samplerMul, substitution_root_pow, Xe, ← pow_add]
  exact polyNorm_Xe hn _

set_option maxHeartbeats 16000000 in
/-- **A PRF sampler.** Member `o` (`M` members) encodes, at `q`, the sum of the `W` branches of
its sampling LUT on `X^(s ζ_o)` plus the branches of the binary offset on `X^(s zoff)`; it is
placed at `X^(pl o)`, the members are summed and the sum is moved down by `p` once. With the LUT
values `V z = Σ_i σ_i(X^(s z)) C_i`, the result encodes `Δ' Σ_o X^(pl o) (V ζ_o + V zoff)`. -/
theorem sampler_carrier_gen (hn : 1 < n) {q q' p m' M W : Nat} (hq : q = p * q') (hq1 : 1 < q)
    (hq' : 1 < q') (hp : 0 < p) {sm sp Δ' : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    (sigma : Fin W → Int) (hodd : ∀ i, Odd (sigma i).toNat)
    (fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q' n 1 m')
    (pl : Fin M → Nat) (hpl : ∀ o, pl o < 2 * n) (s : Nat) (ζ : Fin M → Nat) (zoff : Nat)
    {Ks : Fin M → Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {cs : Fin M → Fin 2 → ExactMatrix q n 1 m'}
    {Bm : Nat}
    (hmem : ∀ o, Rel sm (Ks o)
      (fun j ↦ (∑ i : Fin W, branchTargetG (q := q) (p := p) sigma hodd (Xe (s * ζ o)) sp fam i j) +
        ∑ i : Fin W, branchTargetG (q := q) (p := p) sigma hodd (Xe (s * zoff)) sp fam i j)
      (cs o) Bm)
    {gq : ExactMatrix q' n 1 m'} {C : Fin W → ExactPoly q' n} {BS : Nat}
    (hswitch : ∀ i : Fin W, ∃ e : Fin 2 → ErrorMatrix n 1 m', (∀ j, CoeffBound (e j) BS) ∧ ∀ j,
      reducePoly q' n (substitution (sigma i).toNat (hodd i) sp) • fam 0 j i + fam 1 j i =
        (C i * reducePoly q' n (Δ' * unitPair sp j)) • gq + reduceMatrix q' n 1 m' (e j))
    (V : Nat → ErrorPoly n)
    (hV : ∀ z, ∑ i, reducePoly q' n (substitution (sigma i).toNat (hodd i) (Xe (s * z))) * C i =
      reducePoly q' n (V z))
    {basis : List Nat} {K' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    {R' : Fin 2 → ExactMatrix q' n 1 m'}
    (hK' : ∀ k j, blockModSwitchRuns basis 1 (∑ o, multiplyMonomial (Ks o k j) (pl o : Int))
      (K' k j))
    (hR' : ∀ j, blockModSwitchRuns basis 1 (∑ o, multiplyMonomial (cs o j) (pl o : Int)) (R' j)) :
    Rel sm K' (fun j ↦ reducePoly q' n (Δ' * (∑ o, Xe (n := n) (pl o) * (V (ζ o) + V zoff)) *
      unitPair sp j) • gq) R' ((M * Bm + (n + 2) * (p / 2)) / p + M * (2 * W) * (n * 1 * BS)) := by
  have hn0 : 0 < n := by omega
  -- Each member placed at `X^(pl o)`.
  have hmem' : ∀ o ∈ (Finset.univ : Finset (Fin M)), Rel sm
      (fun a j ↦ multiplyMonomial (Ks o a j) (pl o : Int))
      (fun j ↦ reducePoly q n (Xe (n := n) (pl o)) •
        ((∑ i : Fin W, branchTargetG (q := q) (p := p) sigma hodd (Xe (s * ζ o)) sp fam i j) +
          ∑ i : Fin W, branchTargetG (q := q) (p := p) sigma hodd (Xe (s * zoff)) sp fam i j))
      (fun j ↦ multiplyMonomial (cs o j) (pl o : Int)) Bm :=
    fun o _ ↦ rel_monomial_gen hn (hmem o) (hpl o)
  have hsum := Rel.sum Finset.univ hmem'
  have hkey : (∑ o, fun a j ↦ multiplyMonomial (Ks o a j) (pl o : Int)) =
      fun a j ↦ ∑ o, multiplyMonomial (Ks o a j) (pl o : Int) := by
    funext a j
    simp only [Finset.sum_apply]
  have hrow : (∑ o, fun j ↦ multiplyMonomial (cs o j) (pl o : Int)) =
      fun j ↦ ∑ o, multiplyMonomial (cs o j) (pl o : Int) := by
    funext j
    simp only [Finset.sum_apply]
  rw [hkey, hrow] at hsum
  -- The summed target as one family of branches.
  have hsum' := hsum.target_eq (t' := fun j ↦ ∑ x : Fin M × Fin 2 × Fin W, scaleUpMatrix q p
      (reducePoly q' n (samplerMul sigma hodd pl s ζ zoff x) •
        (reducePoly q' n (substitution (sigma x.2.2).toNat (hodd x.2.2) sp) • fam 0 j x.2.2 +
          fam 1 j x.2.2))) (by
    funext j
    simp only [Finset.sum_apply, Fintype.sum_prod_type, Fin.sum_univ_two, samplerMul,
      branchTargetG]
    refine Finset.sum_congr rfl fun o _ ↦ ?_
    rw [smul_add, Finset.smul_sum, Finset.smul_sum]
    simp only [smul_scaleUpMatrix_gen hn0 hq hq', smul_smul, ← map_mul]
    rfl)
  choose eS heS hS using hswitch
  have hfin := lut_finish_sum_gen (ι := Fin M × Fin 2 × Fin W) hn0 hq hq1 hq' hp hsm
    (samplerMul sigma hodd pl s ζ zoff)
    (fun x ↦ substitution (sigma x.2.2).toNat (hodd x.2.2) sp)
    (fun x ↦ polyNorm_samplerMul hn sigma hodd pl s ζ zoff x)
    (bsw := fun x k j ↦ fam k j x.2.2) hsum' (g' := gq) (C := fun x ↦ C x.2.2)
    (Δ' := Δ') (fun x j ↦ hS x.2.2 j) (fun x j ↦ heS x.2.2 j)
    (ν := ∑ o, Xe (n := n) (pl o) * (V (ζ o) + V zoff)) (by
      simp only [Fintype.sum_prod_type, Fin.sum_univ_two, samplerMul, map_sum, map_mul, map_add]
      refine Finset.sum_congr rfl fun o _ ↦ ?_
      simp only [mul_assoc, ← Finset.mul_sum, ← mul_add]
      congr 1
      rw [← hV, ← hV]
      rfl)
    hK' hR'
  refine hfin.mono (le_of_eq ?_)
  simp [Finset.sum_const, Finset.card_univ, Fintype.card_prod, Fintype.card_fin]

/-! ## The smudge -/

/-- The smudge of `C` columns of `d` digits: `Σ_o X^⌊o/d⌋ w_(o mod d) V_o`. -/
noncomputable def smudgeValueG (C d : Nat) (hd : 0 < d) (w : Fin d → Int)
    (V : Fin (C * d) → ErrorPoly n) : ErrorPoly n :=
  ∑ o : Fin (C * d), Xe (n := n) (o / d) * (((w ⟨o % d, Nat.mod_lt _ hd⟩ : Int) : ErrorPoly n) * V o)

/-- **One smudge member**: a LUT output `V` against the full gadget `G` of an exact layout is
finalized with the digit weight `w (o mod d)` (the decomposition `H'` of the constant) and placed
at `X^⌊o/d⌋`; its bound grows by `m (base / 2)`. -/
theorem smudge_member_gen (hn : 1 < n) {q m C d : Nat} (hq1 : 1 < q) (hd : 0 < d)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm sp V : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {c : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (h : Rel sm A (fun j ↦ reducePoly q n (V * unitPair sp j) • G) c B) (w : Fin d → Int)
    (o : Fin (C * d)) (hplace : o / d < 2 * n)
    {K : ExactMatrix q n 1 1} (hK : K = matrixPolynomial [w ⟨o % d, Nat.mod_lt _ hd⟩])
    {H : ExactMatrix q n m 1} (hH : gadgetDecomposeRuns backend base digits K H)
    {bound : Nat} {H' : ExactMatrix q n m 1} (hH' : compactCenteredRebaseRuns bound H H') :
    Rel sm (fun a j ↦ multiplyMonomial (matrixMul (A a j) H') ((o / d : Nat) : Int))
      (fun j ↦ reducePoly q n (Xe (n := n) (o / d)) •
        reducePoly q n (((w ⟨o % d, Nat.mod_lt _ hd⟩ : Int) : ErrorPoly n) * V * unitPair sp j) •
          (1 : ExactMatrix q n 1 1))
      (fun j ↦ multiplyMonomial (matrixMul (c j) H') ((o / d : Nat) : Int))
      (m * B * (base.toNat / 2)) :=
  rel_monomial_gen hn (h.finalizeCoefficient hq1 (by omega) layout hlayout hexact hG _ hK hH hH')
    hplace

/-- **The smudge chain**: the `C d` placed members sum to an encoding of the smudge with bound
`C d Bm`. -/
theorem smudge_chain_gen {q C d : Nat} (hd : 0 < d) {sm sp : ErrorPoly n} (w : Fin d → Int)
    (V : Fin (C * d) → ErrorPoly n)
    {K : Fin (C * d) → Fin 2 → Fin 2 → ExactMatrix q n 1 1}
    {c : Fin (C * d) → Fin 2 → ExactMatrix q n 1 1} {Bm : Nat}
    (hmem : ∀ o, Rel sm (K o) (fun j ↦ reducePoly q n (Xe (n := n) (o / d)) •
      reducePoly q n (((w ⟨o % d, Nat.mod_lt _ hd⟩ : Int) : ErrorPoly n) * V o * unitPair sp j) •
        (1 : ExactMatrix q n 1 1)) (c o) Bm) :
    Rel sm (∑ o, K o) (fun j ↦ reducePoly q n (smudgeValueG C d hd w V * unitPair sp j) •
      (1 : ExactMatrix q n 1 1)) (∑ o, c o) ((C * d) * Bm) := by
  have hsum := Rel.sum Finset.univ (fun o _ ↦ hmem o)
  refine (hsum.mono (le_of_eq (by simp))).target_eq ?_
  funext j
  rw [Finset.sum_apply, smudgeValueG, Finset.sum_mul, map_sum, Finset.sum_smul]
  refine Finset.sum_congr rfl fun o _ ↦ ?_
  rw [smul_smul, ← map_mul]
  congr 2
  ring

/-! ## One branch of the loop, from the atoms of the generated body -/

set_option maxHeartbeats 64000000 in
/-- **One iteration of the LUT loop** (`lut.round-loop`, the body of a `seq[W-1]` child) from the
atoms of the generated body: the input encoding routed by `σ_i` (`ringAutomorphismRuns`), the
carried switching ciphertext of branch `i` decomposed (`D`), switched, moved down (`Asw`) and
decomposed again (`E`), the helper of branch `i` (`h3`, `Ad`) acting through `E`; the next state
adds `Y + X` to the row and `K` to the key and carries branch `i + 1` of the switching families.
This is the step hypothesis of `lut_loop_gen` for any `i`. -/
theorem lut_step_gen (hn : 0 < n) {log : Nat} (hpow : n = 2 ^ log)
    {q q' p m m' bound W : Nat} (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q')
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    (sigma : Fin W → Int) (hodd : ∀ i, Odd (sigma i).toNat)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix q' n 1 m} {g : ExactMatrix q n 1 m}
    (hg' : gadgetMatrixRuns backend base digits g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix q n 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (hinput : Rel sm A (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) c B)
    (fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q' n 1 m') (i : Nat) (hi : i + 1 < W)
    {cur nxt : LoopStateG n q q' m'} {Tc : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hcur : Rel sm cur.key Tc cur.row Bs)
    (hsw : cur.switch = fun k j ↦ fam k j ⟨i, by omega⟩)
    {h3 : Fin 2 → ExactMatrix q n 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {Bh : Nat}
    (hhelp : Rel sm Ad (fun j ↦ reducePoly q n ((p : Int) *
        unitPair (substitution (sigma ⟨i, by omega⟩).toNat (hodd ⟨i, by omega⟩) sm) j) • g) h3 Bh)
    {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q n 1 m}
    (hr0 : ringAutomorphismRuns (sigma ⟨i, by omega⟩) (c 0) rc0)
    (hr1 : ringAutomorphismRuns (sigma ⟨i, by omega⟩) (A 0 0) rA00)
    (hr2 : ringAutomorphismRuns (sigma ⟨i, by omega⟩) (A 1 0) rA10)
    (hr3 : ringAutomorphismRuns (sigma ⟨i, by omega⟩) (c 1) rc1)
    (hr4 : ringAutomorphismRuns (sigma ⟨i, by omega⟩) (A 0 1) rA01)
    (hr5 : ringAutomorphismRuns (sigma ⟨i, by omega⟩) (A 1 1) rA11)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows rc0 rA00 l01) (hl : concatRows l01 rA10 l)
    (hr01 : concatRows rc1 rA01 r01) (hr : concatRows r01 rA11 r) (hboth : concatColumns l r both)
    {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m'}
    (hXb : ∀ j, concatRows (cur.switch 0 j) (cur.switch 1 j) (Xb j))
    {D' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {basis : List Nat} {Asw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    (hAsw : ∀ k j, blockModSwitchRuns basis 1 (M k j) (Asw k j))
    {Z : Fin 2 → ExactMatrix q' n (1 + 1) m'} (hZ : ∀ j, concatRows (Asw 0 j) (Asw 1 j) (Z j))
    {E' : Fin 2 → ExactMatrix q' n (m + m) m'}
    (hE' : ∀ j, gadgetDecomposeRuns backend base digits (Z j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m + m) m'} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {hl01' hr01' : ExactMatrix q n (1 + 1) m} {hl' hr' : ExactMatrix q n (1 + 1 + 1) m}
    {H : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hhl01 : concatRows (h3 0) (Ad 0 0) hl01') (hhl : concatRows hl01' (Ad 1 0) hl')
    (hhr01 : concatRows (h3 1) (Ad 0 1) hr01') (hhr : concatRows hr01' (Ad 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {Y X : Fin 2 → ExactMatrix q n 1 m'} {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul H (E j)) 0 1 0 m' (Y j))
    (hX : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (X j))
    (hK0 : ∀ j, sliceMatrix (matrixMul H (E j)) 1 2 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixMul H (E j)) 2 3 0 m' (K 1 j))
    (hrow : ∀ j, nxt.row j = cur.row j + (Y j + X j))
    (hkey : ∀ a j, nxt.key a j = cur.key a j + K a j)
    (hnsw : nxt.switch = fun k j ↦ fam k j ⟨i + 1, hi⟩) :
    Rel sm nxt.key (Tc + lutBranchTargetG (q := q) (p := p) sigma hodd μ sp fam i) nxt.row
        (Bs + lutBranchBoundG n m bound p B Bh) ∧
      nxt.switch = fun k j ↦ fam k j ⟨i + 1, hi⟩ := by
  refine ⟨?_, hnsw⟩
  have hb := lut_branch (q := q) (q' := q') (p := p) (m := m) (m' := m') (log := log)
    (bound := bound) (index := sigma ⟨i, by omega⟩) hn hpow hq hq1 hq' layout hlayout hexact hg'
    hg hsm hinput hr0 hr1 hr2 hr3 hr4 hr5 hl01 hl hr01 hr hboth (bsw := cur.switch) hXb hD' hD
    hM0 hM1 hAsw hZ hE' hE (h := h3) (hodd _) hhelp hhl01 hhl hhr01 hhr hH hY hX hK0 hK1
  rw [hsw] at hb
  have hsum := hcur.add hb
  have hk : nxt.key = cur.key + K := funext fun a ↦ funext fun j ↦ hkey a j
  have hr : nxt.row = cur.row + fun j ↦ Y j + X j := funext fun j ↦ hrow j
  rw [hk, hr]
  refine hsum.target_eq ?_
  funext j
  simp only [Pi.add_apply, lutBranchTargetG, dif_pos (show i < W by omega)]
  rfl

/-! ## The smudge member as exported: digit weight by the fold, placement at any `X^e` -/

/-- One smudge member with its weight given as `matrixPolynomial [w k]` (the form
`smudge_member_place_gen` derives from the fold). -/
theorem smudge_member_weight_gen (hn : 1 < n) {q m : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm sp V : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {c : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (h : Rel sm A (fun j ↦ reducePoly q n (V * unitPair sp j) • G) c B) (wk : Int)
    {K : ExactMatrix q n 1 1} (hK : K = matrixPolynomial [wk])
    {H : ExactMatrix q n m 1} (hH : gadgetDecomposeRuns backend base digits K H)
    {bound : Nat} {H' : ExactMatrix q n m 1} (hH' : compactCenteredRebaseRuns bound H H')
    {e : Nat} (he : e < 2 * n) :
    Rel sm (fun a j ↦ multiplyMonomial (matrixMul (A a j) H') (e : Int))
      (fun j ↦ reducePoly q n (Xe (n := n) e) •
        reducePoly q n ((wk : ErrorPoly n) * V * unitPair sp j) • (1 : ExactMatrix q n 1 1))
      (fun j ↦ multiplyMonomial (matrixMul (c j) H') (e : Int))
      (m * B * (base.toNat / 2)) :=
  rel_monomial_gen hn (h.finalizeCoefficient hq1 (by omega) layout hlayout hexact hG _ hK hH hH') he

/-- **One smudge member** (`smudge.member`) as the export computes it: the LUT output `V`
(against the full gadget `G` of an exact layout) is finalized with the weight of digit `k`, read as
column `k` of the `concatColumns` fold `x` of the digit weights (`RowEntries`, built by
`rowEntries_one`/`rowEntries_concat`, one step per digit), and placed at `X^e` (the export's
`e = ⌊o / d⌋ mod N`, `k = o mod d`; `fdiv_fmod_ofNat`, `fmod_ofNat`). -/
theorem smudge_member_place_gen (hn : 1 < n) {q m d : Nat} (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {G : ExactMatrix q n 1 m} (hG : gadgetMatrixRuns backend base digits G)
    {sm sp V : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {c : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (h : Rel sm A (fun j ↦ reducePoly q n (V * unitPair sp j) • G) c B) (w : Nat → Int)
    {x : ExactMatrix q n 1 d} (hx : RowEntries x fun c ↦ ((w c : Int) : ExactPoly q n))
    {k : Nat} (hk : k < d) {K : ExactMatrix q n 1 1}
    (hK : sliceMatrix x 0 1 (k : Int) (1 + (k : Int)) K)
    {H : ExactMatrix q n m 1} (hH : gadgetDecomposeRuns backend base digits K H)
    {bound : Nat} {H' : ExactMatrix q n m 1} (hH' : compactCenteredRebaseRuns bound H H')
    {e : Nat} (he : e < 2 * n) :
    Rel sm (fun a j ↦ multiplyMonomial (matrixMul (A a j) H') (e : Int))
      (fun j ↦ reducePoly q n (Xe (n := n) e) •
        reducePoly q n (((w k : Int) : ErrorPoly n) * V * unitPair sp j) •
          (1 : ExactMatrix q n 1 1))
      (fun j ↦ multiplyMonomial (matrixMul (c j) H') (e : Int))
      (m * B * (base.toNat / 2)) := by
  have hK' : K = matrixPolynomial [w k] := by
    rw [slice_rowEntries hx hk hK]
    funext r c'
    rw [matrixPolynomial_single]
  exact smudge_member_weight_gen hn hq1 layout hlayout hexact hG h (w k) hK' hH hH' he

/-! ## Composing the carriers -/

/-- **The chain composition**: the mask, fresh and smudge carriers (any three encodings under the
same mask) add to an encoding of the summed targets with the summed bounds. -/
theorem chain3_gen {q m : Nat} {sm : ErrorPoly n}
    {A₁ A₂ A₃ : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {t₁ t₂ t₃ c₁ c₂ c₃ : Fin 2 → ExactMatrix q n 1 m}
    {B₁ B₂ B₃ : Nat} (h₁ : Rel sm A₁ t₁ c₁ B₁) (h₂ : Rel sm A₂ t₂ c₂ B₂) (h₃ : Rel sm A₃ t₃ c₃ B₃) :
    Rel sm (A₁ + A₂ + A₃) (t₁ + t₂ + t₃) (c₁ + c₂ + c₃) (B₁ + B₂ + B₃) :=
  (h₁.add h₂).add h₃

/-- `chain3_gen` on six-matrix states (the export adds the carriers component by component). -/
theorem relT6_add {q m : Nat} {sm : ErrorPoly n} (L : Layout6)
    {t₁ t₂ : Fin 2 → ExactMatrix q n 1 m} {x₁ x₂ : T6 (ExactMatrix q n 1 m)} {B₁ B₂ : Nat}
    (h₁ : RelT6 sm L t₁ x₁ B₁) (h₂ : RelT6 sm L t₂ x₂ B₂) :
    RelT6 sm L (t₁ + t₂) (x₁ + x₂) (B₁ + B₂) := by
  have := RelT6.sum L (Finset.univ : Finset (Fin 2)) (t := ![t₁, t₂]) (f := ![x₁, x₂])
    (B := ![B₁, B₂]) (fun i _ ↦ by fin_cases i; exacts [h₁, h₂])
  simpa [Fin.sum_univ_two] using this

end RefreshTopProof.Gen
