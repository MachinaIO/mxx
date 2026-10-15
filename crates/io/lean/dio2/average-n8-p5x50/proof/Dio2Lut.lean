import ExponentLut

/-!
# LUT branches without a modulus drop

A dio2 chain keeps the modulus `Q_0`, so a LUT branch keeps its switched key where refresh_top's
(`lut_branch`) moves it down: `lut_branch_flat` is `lut_branch` with `p = 1`, whose ModDown
remainder is zero.
-/

namespace Dio2Proof

open Mxx.Primitives MxxRuntime MxxExponentLut

variable {n : Nat}

set_option maxHeartbeats 1000000 in
/-- One LUT branch at one modulus: the routed input `σ(c)` switched by the decomposition of the
switching ciphertexts `bsw`, plus the helper acted on by the decomposition of the switched key. -/
theorem lut_branch_flat {q m m' log bound : Nat} (hn : 0 < n) (hpow : n = 2 ^ log)
    (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {g' : ExactMatrix q n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {c : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly q n (((1 : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {index : Int} {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q n 1 m}
    (hr0 : ringAutomorphismRuns index (c 0) rc0) (hr1 : ringAutomorphismRuns index (A 0 0) rA00)
    (hr2 : ringAutomorphismRuns index (A 1 0) rA10) (hr3 : ringAutomorphismRuns index (c 1) rc1)
    (hr4 : ringAutomorphismRuns index (A 0 1) rA01) (hr5 : ringAutomorphismRuns index (A 1 1) rA11)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows rc0 rA00 l01) (hl : concatRows l01 rA10 l)
    (hr01 : concatRows rc1 rA01 r01) (hr : concatRows r01 rA11 r) (hboth : concatColumns l r both)
    {bsw : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {Xb : Fin 2 → ExactMatrix q n (1 + 1) m'}
    (hXb : ∀ j, concatRows (bsw 0 j) (bsw 1 j) (Xb j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {Z : Fin 2 → ExactMatrix q n (1 + 1) m'} (hZ : ∀ j, concatRows (M 0 j) (M 1 j) (Z j))
    {E' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hE' : ∀ j, gadgetDecomposeRuns backend base digits (Z j) (E' j))
    {E : Fin 2 → ExactMatrix q n (m + m) m'} (hE : ∀ j, compactCenteredRebaseRuns bound (E' j) (E j))
    {h : Fin 2 → ExactMatrix q n 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {Bh : Nat}
    (hodd : Odd index.toNat)
    (hhelp : Rel sm Ad (fun j ↦ reducePoly q n (((1 : Nat) : Int) *
      unitPair (substitution index.toNat hodd sm) j) • g) h Bh)
    {hl01' hr01' : ExactMatrix q n (1 + 1) m} {hl' hr' : ExactMatrix q n (1 + 1 + 1) m}
    {H : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hhl01 : concatRows (h 0) (Ad 0 0) hl01') (hhl : concatRows hl01' (Ad 1 0) hl')
    (hhr01 : concatRows (h 1) (Ad 0 1) hr01') (hhr : concatRows hr01' (Ad 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {Y X : Fin 2 → ExactMatrix q n 1 m'} {K : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hY : ∀ j, sliceMatrix (matrixMul H (E j)) 0 1 0 m' (Y j))
    (hX : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (X j))
    (hK0 : ∀ j, sliceMatrix (matrixMul H (E j)) 1 2 0 m' (K 0 j))
    (hK1 : ∀ j, sliceMatrix (matrixMul H (E j)) 2 3 0 m' (K 1 j)) :
    Rel sm K (fun j ↦ reducePoly q n (substitution index.toNat hodd μ) •
        (reducePoly q n (substitution index.toNat hodd sp) • bsw 0 j + bsw 1 j))
      (fun j ↦ Y j + X j)
      ((m + m) * n * Bh * bound + (m + m) * n * B * bound) := by
  have hq : q = 1 * q := (one_mul q).symm
  have hcop : n.Coprime index.toNat := by rw [hpow]; exact coprime_of_odd_pow_two hodd
  -- The routed blocks.
  have hroute : ∀ {x y : ExactMatrix q n 1 m}, ringAutomorphismRuns index x y →
      y = substituteMatrix index.toNat hodd x := by
    intro x y hxy
    obtain ⟨_, hy⟩ := ringAutomorphismRuns_eq hq1 hn hxy
    exact hy
  have hsub := henc.substitute hn hodd hcop
  -- Rows of the stacked products.
  have hrows := fun j ↦ stacked_row hl01 hl hr01 hr hboth (D j)
  have hhrows := fun j ↦ stacked_row hhl01 hhl hhr01 hhr hH (E j)
  have hslice : ∀ {rows cols : Nat} {x : ExactMatrix q n rows cols} {out : ExactMatrix q n 1 cols}
      (k : Fin rows), sliceMatrix x k.val (k.val + 1) 0 cols out → out = rowOf x k :=
    fun k hk ↦ sliceMatrix_row k hk
  have hσc : (fun i ↦ substituteMatrix index.toNat hodd (c i)) = ![rc0, rc1] := by
    rw [hroute hr0, hroute hr3]; exact (pair_eta _).symm
  have hσA0 : (fun i ↦ substituteMatrix index.toNat hodd (A 0 i)) = ![rA00, rA01] := by
    rw [hroute hr1, hroute hr4]; exact (pair_eta _).symm
  have hσA1 : (fun i ↦ substituteMatrix index.toNat hodd (A 1 i)) = ![rA10, rA11] := by
    rw [hroute hr2, hroute hr5]; exact (pair_eta _).symm
  -- The switching action on the routed encoding.
  choose d hd using fun j ↦ (hD j).2.2.2
  have hDd : ∀ j, D j = reduceMatrix q n (m + m) m' (d j) := fun j ↦ (hd j).1
  have hact := hsub.action hn d (fun j ↦ (hd j).2)
  simp only [← hDd] at hact
  have hXe : ∀ j, X j = pairMul (fun i ↦ substituteMatrix index.toNat hodd (c i)) (D j) := by
    intro j
    rw [hslice (rows := 1 + 1 + 1) 0 (hX j), hσc]
    exact (hrows j).1
  have hMe : ∀ k j, M k j = pairMul (fun i ↦ substituteMatrix index.toNat hodd (A k i)) (D j) := by
    intro k j
    match k with
    | 0 =>
      rw [hslice (rows := 1 + 1 + 1) 1 (hM0 j), hσA0]
      exact (hrows j).2.1
    | 1 =>
      rw [hslice (rows := 1 + 1 + 1) 2 (hM1 j), hσA1]
      exact (hrows j).2.2
  -- The helper action.
  choose e he using fun j ↦ (hE j).2.2.2
  have hEe : ∀ j, E j = reduceMatrix q n (m + m) m' (e j) := fun j ↦ (he j).1
  have hhact := hhelp.action hn e (fun j ↦ (he j).2)
  simp only [← hEe] at hhact
  have hYe : ∀ j, Y j = pairMul h (E j) := by
    intro j
    rw [hslice (rows := 1 + 1 + 1) 0 (hY j), ← pair_eta h]
    exact (hhrows j).1
  have hKe : ∀ k j, K k j = pairMul (Ad k) (E j) := by
    intro k j
    match k with
    | 0 =>
      rw [hslice (rows := 1 + 1 + 1) 1 (hK0 j), ← pair_eta (Ad 0)]
      exact (hhrows j).2.1
    | 1 =>
      rw [hslice (rows := 1 + 1 + 1) 2 (hK1 j), ← pair_eta (Ad 1)]
      exact (hhrows j).2.2
  -- The routed target after the switching action.
  have htc : ∀ j, pairMul (fun i ↦ substituteMatrix index.toNat hodd
      (reducePoly q n (((1 : Nat) : Int) * μ * unitPair sp i) • g)) (D j) =
      scaleUpMatrix q 1 (reducePoly q n (substitution index.toNat hodd μ) •
        (reducePoly q n (substitution index.toNat hodd sp) • bsw 0 j + bsw 1 j)) := by
    intro j
    simp only [substituteMatrix_target hn hg' hg]
    have hgad := gadget_action_scaleUp hq hq1 hn layout hlayout hexact hg' hg (hD' j) (hD j)
      (fun i ↦ substitution index.toNat hodd μ * substitution index.toNat hodd (unitPair sp i))
    simp only [← mul_assoc] at hgad
    rw [hgad, (concatRows_two (hXb j)).1, (concatRows_two (hXb j)).2]
    congr 1
    simp only [unitPair, Matrix.cons_val_zero, Matrix.cons_val_one, map_one, map_mul, mul_one,
      smul_add, smul_smul]
  -- The helper target after its action.
  have hth : ∀ j, pairMul (fun i ↦ reducePoly q n (((1 : Nat) : Int) *
      unitPair (substitution index.toNat hodd sm) i) • g) (E j) =
      scaleUpMatrix q 1 (reducePoly q n (substitution index.toNat hodd sm) • M 0 j + M 1 j) := by
    intro j
    rw [gadget_action_scaleUp hq hq1 hn layout hlayout hexact hg' hg (hE' j) (hE j),
      (concatRows_two (hZ j)).1, (concatRows_two (hZ j)).2]
    simp [unitPair]
  -- No ModDown: the switched key is kept.
  have hsw : ∀ k j, ∃ r : ErrorMatrix n 1 m', CoeffBound r (1 / 2) ∧
      scaleUpMatrix q 1 (M k j) = pairMul (fun i ↦ substituteMatrix index.toNat hodd (A k i)) (D j) -
        reduceMatrix q n 1 m' r := by
    intro k j
    refine ⟨0, fun _ _ _ ↦ by simp, ?_⟩
    rw [scaleUpMatrix_one hq1 hn, hMe k j]
    funext row column
    simp [reduceMatrix]
  have hsmσ : polyNorm (substitution index.toNat hodd sm) ≤ 1 :=
    (polyNorm_substitution_le hn hodd hcop sm).trans hsm
  have hbranch := Rel.lutBranch hn hq hq1 hsmσ hact hsw hhact hth
  simp only [htc, scaleUpMatrix_one hq1 hn] at hbranch
  have hYX : (fun j ↦ Y j + X j) = (fun j ↦ pairMul h (E j)) +
      fun j ↦ pairMul (fun i ↦ substituteMatrix index.toNat hodd (c i)) (D j) := by
    funext j
    rw [Pi.add_apply, hYe j, hXe j]
  have hKfun : K = fun k j ↦ pairMul (Ad k) (E j) := by
    funext k j
    exact hKe k j
  rw [hYX, hKfun]
  exact hbranch.mono (le_of_eq (by simp))

set_option maxHeartbeats 1000000 in
/-- A LUT finished at its own modulus (`lut_finish` without the move down): the summed branches
`Σ_i σ_i(μ) (σ_i(r₀) B^i_{0j} + B^i_{1j})`, where `σ_i(r₀) B^i_{0j} + B^i_{1j} = (C_i Δ' r_j) g'
+ e^i_j` and `Σ_i σ_i(μ) C_i = ν`, encode `Δ' ν` with gadget `g'`. -/
theorem lut_finish_flat {q m' W : Nat} (hn : 0 < n) {sm sp μ ν Δ' : ErrorPoly n}
    (σ : Fin W → ErrorPoly n → ErrorPoly n) (hσμ : ∀ i, polyNorm (σ i μ) ≤ 1)
    {bsw : Fin W → Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    {Ksum : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {Rsum : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hsum : Rel sm Ksum (fun j ↦ ∑ i, reducePoly q n (σ i μ) •
      (reducePoly q n (σ i sp) • bsw i 0 j + bsw i 1 j)) Rsum Bs)
    {g' : ExactMatrix q n 1 m'} {C : Fin W → ExactPoly q n} {eB : Fin W → Fin 2 → ErrorMatrix n 1 m'}
    {BB : Nat} (hswitch : ∀ i j, reducePoly q n (σ i sp) • bsw i 0 j + bsw i 1 j =
      (C i * reducePoly q n (Δ' * unitPair sp j)) • g' + reduceMatrix q n 1 m' (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (hwalsh : ∑ i, reducePoly q n (σ i μ) * C i = reducePoly q n ν) :
    Rel sm Ksum (fun j ↦ reducePoly q n (Δ' * ν * unitPair sp j) • g') Rsum
      (Bs + W * (n * 1 * BB)) := by
  let f : Fin 2 → ErrorMatrix n 1 m' := fun j ↦ ∑ i, σ i μ • eB i j
  have htarget : ∀ j, (∑ i, reducePoly q n (σ i μ) •
      (reducePoly q n (σ i sp) • bsw i 0 j + bsw i 1 j)) =
      reducePoly q n (Δ' * ν * unitPair sp j) • g' + reduceMatrix q n 1 m' (f j) := by
    intro j
    simp only [hswitch]
    funext row column
    simp only [Matrix.sum_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul,
      reduceMatrix, f]
    have hterm : ∀ i ∈ (Finset.univ : Finset (Fin W)), reducePoly q n (σ i μ) *
        (C i * reducePoly q n (Δ' * unitPair sp j) * g' row column +
          reducePoly q n (eB i j row column)) =
        (reducePoly q n (σ i μ) * C i) * (reducePoly q n (Δ' * unitPair sp j) * g' row column) +
          reducePoly q n (σ i μ * eB i j row column) := by
      intro i _
      simp only [map_mul]
      ring
    rw [Finset.sum_congr rfl hterm, Finset.sum_add_distrib, ← Finset.sum_mul, hwalsh,
      map_sum]
    simp only [map_mul]
    ring
  have hsum' : Rel sm Ksum (fun j ↦ reducePoly q n (Δ' * ν * unitPair sp j) • g' +
      reduceMatrix q n 1 m' (f j)) Rsum Bs := hsum.target_eq (funext htarget)
  have hf : ∀ j, CoeffBound (f j) (W * (n * 1 * BB)) := by
    intro j
    have := coeffBound_sum Finset.univ (fun i _ ↦ coeffBound_smul hn (hσμ i) (heB i j))
    simpa [Finset.sum_const, Finset.card_univ, Fintype.card_fin] using this
  exact hsum'.absorb hf


/-- Branch `i` of a LUT with automorphisms `sigma` over the switching families `fam`, at one
modulus: `σ_i(μ) (σ_i(r₀) B^i_{0j} + B^i_{1j})`. -/
noncomputable def branchFlat {q m W : Nat} (sigma : Fin W → Int) (hodd : ∀ i, Odd (sigma i).toNat)
    (μ sp : ErrorPoly n) (fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q n 1 m) (i : Fin W) :
    Fin 2 → ExactMatrix q n 1 m :=
  fun j ↦ reducePoly q n (substitution (sigma i).toNat (hodd i) μ) •
    (reducePoly q n (substitution (sigma i).toNat (hodd i) sp) • fam 0 j i + fam 1 j i)

/-- `branchFlat`, zero past the `W` branches (the loop's term). -/
noncomputable def lutBranchFlat {q m W : Nat} (sigma : Fin W → Int)
    (hodd : ∀ i, Odd (sigma i).toNat) (μ sp : ErrorPoly n)
    (fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q n 1 m) (i : Nat) : Fin 2 → ExactMatrix q n 1 m :=
  if h : i < W then branchFlat sigma hodd μ sp fam ⟨i, h⟩ else 0

set_option maxHeartbeats 64000000 in
/-- **A LUT round at one modulus**: the loop's first `L` branches (`hloop`, from `lut_loop_gen`),
the last branch and the finish. If the switching ciphertexts encode the coefficients `C_i` and
`Σ_i σ_i(μ) C_i = ν`, the result encodes `Δ' ν` against `gq`. -/
theorem lut_round_flat (hn : 0 < n) {log : Nat} (hpow : n = 2 ^ log)
    {q m m' bound W L : Nat} (hL : L + 1 = W) (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    (sigma : Fin W → Int) (hodd : ∀ i, Odd (sigma i).toNat)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix q n 1 m} {g : ExactMatrix q n 1 m}
    (hg' : gadgetMatrixRuns backend base digits g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix q n 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (hinput : Rel sm A (fun j ↦ reducePoly q n (((1 : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q n 1 m'}
    {Kl : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {Rl : Fin 2 → ExactMatrix q n 1 m'} {Bl : Nat}
    (hloop : Rel sm Kl (∑ i ∈ Finset.range L, lutBranchFlat sigma hodd μ sp fam i) Rl Bl)
    {h3 : Fin 2 → ExactMatrix q n 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {Bh : Nat}
    (hhelp : Rel sm Ad (fun j ↦ reducePoly q n (((1 : Nat) : Int) *
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
    {Xb : Fin 2 → ExactMatrix q n (1 + 1) m'}
    (hXb : ∀ j, concatRows (fam 0 j ⟨L, by omega⟩) (fam 1 j ⟨L, by omega⟩) (Xb j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {M : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hM0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (M 0 j))
    (hM1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (M 1 j))
    {Z : Fin 2 → ExactMatrix q n (1 + 1) m'} (hZ : ∀ j, concatRows (M 0 j) (M 1 j) (Z j))
    {E' : Fin 2 → ExactMatrix q n (m + m) m'}
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
    {Δ' ν : ErrorPoly n} {gq : ExactMatrix q n 1 m'} {C : Fin W → ExactPoly q n} {BB : Nat}
    (hswitch : ∀ i : Fin W, ∃ e : Fin 2 → ErrorMatrix n 1 m', (∀ j, CoeffBound (e j) BB) ∧ ∀ j,
      reducePoly q n (substitution (sigma i).toNat (hodd i) sp) • fam 0 j i + fam 1 j i =
        (C i * reducePoly q n (Δ' * unitPair sp j)) • gq + reduceMatrix q n 1 m' (e j))
    (hμ : ∀ i, polyNorm (substitution (sigma i).toNat (hodd i) μ) ≤ 1)
    (hwalsh : ∑ i, reducePoly q n (substitution (sigma i).toNat (hodd i) μ) * C i =
      reducePoly q n ν) :
    Rel sm (fun k j ↦ Kl k j + K k j) (fun j ↦ reducePoly q n (Δ' * ν * unitPair sp j) • gq)
      (fun j ↦ Rl j + (Y j + X j))
      (Bl + ((m + m) * n * Bh * bound + (m + m) * n * B * bound) + W * (n * 1 * BB)) := by
  have hbranch := lut_branch_flat (index := sigma ⟨L, by omega⟩) (log := log) (bound := bound)
    hn hpow hq1 layout hlayout hexact hg' hg hsm hinput hr0 hr1 hr2 hr3 hr4 hr5 hl01 hl hr01 hr
    hboth (bsw := fun k j ↦ fam k j ⟨L, by omega⟩) hXb hD' hD hM0 hM1 hZ hE' hE (hodd _) hhelp
    hhl01 hhl hhr01 hhr hH hY hX hK0 hK1
  have hsum := hloop.add hbranch
  have hsum' : Rel sm (fun k j ↦ Kl k j + K k j) (fun j ↦ ∑ i : Fin W,
      reducePoly q n (substitution (sigma i).toNat (hodd i) μ) •
        (reducePoly q n (substitution (sigma i).toNat (hodd i) sp) • fam 0 j i + fam 1 j i))
      (fun j ↦ Rl j + (Y j + X j)) (Bl + ((m + m) * n * Bh * bound + (m + m) * n * B * bound)) := by
    refine hsum.target_eq ?_
    funext j
    subst hL
    rw [Fin.sum_univ_castSucc, Pi.add_apply, Finset.sum_apply, Finset.sum_range]
    refine congrArg₂ (· + ·) (Finset.sum_congr rfl fun i _ ↦ ?_) ?_
    · rw [lutBranchFlat, dif_pos (by omega)]
      rfl
    · rfl
  choose eS heS hS using hswitch
  exact lut_finish_flat (W := W) (q := q) (m' := m') hn (sp := sp) (μ := μ) (Δ' := Δ')
    (fun i ↦ substitution (sigma i).toNat (hodd i)) hμ (bsw := fun i k j ↦ fam k j i) hsum'
    (g' := gq) (C := C) (fun i j ↦ hS i j) (fun i j ↦ heS i j) hwalsh

/-- Row `k` of a matrix is its slice `k .. k + 1`. -/
theorem sliceMatrix_rowOf {q rows columns : Nat} (x : ExactMatrix q n rows columns) (k : Fin rows) :
    sliceMatrix x (k.val : Int) ((k.val : Int) + 1) 0 (columns : Int) (rowOf x k) := by
  refine ⟨by omega, by omega, by have := k.isLt; simp only [Int.ofNat_eq_natCast]; omega, le_refl _,
    by omega, by simp, by simp, by simp, fun row column hRow hColumn ↦ ?_⟩
  have hr : row = 0 := Subsingleton.elim _ _
  subst hr
  simp only [rowOf]
  congr 1 <;> apply Fin.ext <;> simp

set_option maxHeartbeats 64000000 in
/-- **A LUT round at one modulus, read at component `1`.** The generated loop's first `L`
branches (`hloop`, both components) and the last branch's component `1` only (the graph computes
nothing else of it when only that component is read): the summed row `R_l 1 + (Y + X)` is
`s̄ K₀ + K₁ - Δ' ν g_q` plus an error, with `K_k = K_l k 1 + row (k + 1) of H E`. -/
theorem lut_round_flat_last1 (hn : 0 < n) {log : Nat} (hpow : n = 2 ^ log)
    {q m m' bound W L : Nat} (hL : L + 1 = W) (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    (sigma : Fin W → Int) (hodd : ∀ i, Odd (sigma i).toNat)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix q n 1 m} {g : ExactMatrix q n 1 m}
    (hg' : gadgetMatrixRuns backend base digits g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix q n 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (hinput : Rel sm A (fun j ↦ reducePoly q n (((1 : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q n 1 m'}
    {Kl : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {Rl : Fin 2 → ExactMatrix q n 1 m'} {Bl : Nat}
    (hloop : Rel sm Kl (∑ i ∈ Finset.range L, lutBranchFlat sigma hodd μ sp fam i) Rl Bl)
    {h3 : Fin 2 → ExactMatrix q n 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {Bh : Nat}
    (hhelp : Rel sm Ad (fun j ↦ reducePoly q n (((1 : Nat) : Int) *
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
    {Xb : ExactMatrix q n (1 + 1) m'}
    (hXb : concatRows (fam 0 1 ⟨L, by omega⟩) (fam 1 1 ⟨L, by omega⟩) Xb)
    {D' : ExactMatrix q n (m + m) m'} (hD' : gadgetDecomposeRuns backend base digits Xb D')
    {D : ExactMatrix q n (m + m) m'} (hD : compactCenteredRebaseRuns bound D' D)
    {M0 M1 : ExactMatrix q n 1 m'}
    (hM0 : sliceMatrix (matrixMul both D) 1 2 0 m' M0)
    (hM1 : sliceMatrix (matrixMul both D) 2 3 0 m' M1)
    {Z : ExactMatrix q n (1 + 1) m'} (hZ : concatRows M0 M1 Z)
    {E' : ExactMatrix q n (m + m) m'} (hE' : gadgetDecomposeRuns backend base digits Z E')
    {E : ExactMatrix q n (m + m) m'} (hE : compactCenteredRebaseRuns bound E' E)
    {hl01' hr01' : ExactMatrix q n (1 + 1) m} {hl' hr' : ExactMatrix q n (1 + 1 + 1) m}
    {H : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hhl01 : concatRows (h3 0) (Ad 0 0) hl01') (hhl : concatRows hl01' (Ad 1 0) hl')
    (hhr01 : concatRows (h3 1) (Ad 0 1) hr01') (hhr : concatRows hr01' (Ad 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {Y X : ExactMatrix q n 1 m'}
    (hY : sliceMatrix (matrixMul H E) 0 1 0 m' Y)
    (hX : sliceMatrix (matrixMul both D) 0 1 0 m' X)
    {Δ' ν : ErrorPoly n} {gq : ExactMatrix q n 1 m'} {C : Fin W → ExactPoly q n} {BB : Nat}
    (hswitch : ∀ i : Fin W, ∃ e : ErrorMatrix n 1 m', CoeffBound e BB ∧
      reducePoly q n (substitution (sigma i).toNat (hodd i) sp) • fam 0 1 i + fam 1 1 i =
        (C i * reducePoly q n Δ') • gq + reduceMatrix q n 1 m' e)
    (hμ : ∀ i, polyNorm (substitution (sigma i).toNat (hodd i) μ) ≤ 1)
    (hwalsh : ∑ i, reducePoly q n (substitution (sigma i).toNat (hodd i) μ) * C i =
      reducePoly q n ν) :
    ∃ e : ErrorMatrix n 1 m', CoeffBound e
        (Bl + ((m + m) * n * Bh * bound + (m + m) * n * B * bound) + W * (n * 1 * BB)) ∧
      matrixAdd (Rl 1) (matrixAdd Y X) =
        reducePoly q n sm • (Kl 0 1 + rowOf (matrixMul H E) 1) +
          (Kl 1 1 + rowOf (matrixMul H E) 2) - reducePoly q n (Δ' * ν) • gq +
          reduceMatrix q n 1 m' e := by
  -- The last branch, its component-1 data duplicated to both components.
  have hbranch := lut_branch_flat (index := sigma ⟨L, by omega⟩) (log := log) (bound := bound)
    hn hpow hq1 layout hlayout hexact hg' hg hsm hinput hr0 hr1 hr2 hr3 hr4 hr5 hl01 hl hr01 hr
    hboth (bsw := fun k _ ↦ fam k 1 ⟨L, by omega⟩) (Xb := fun _ ↦ Xb) (fun _ ↦ hXb)
    (D' := fun _ ↦ D') (fun _ ↦ hD') (D := fun _ ↦ D) (fun _ ↦ hD)
    (M := fun k _ ↦ ![M0, M1] k) (fun _ ↦ hM0) (fun _ ↦ hM1)
    (Z := fun _ ↦ Z) (fun _ ↦ hZ) (E' := fun _ ↦ E') (fun _ ↦ hE') (E := fun _ ↦ E) (fun _ ↦ hE)
    (hodd _) hhelp hhl01 hhl hhr01 hhr hH
    (Y := fun _ ↦ Y) (X := fun _ ↦ X)
    (K := fun k _ ↦ ![rowOf (matrixMul H E) 1, rowOf (matrixMul H E) 2] k) (fun _ ↦ hY) (fun _ ↦ hX)
    (fun _ ↦ by simpa using sliceMatrix_rowOf (matrixMul H E) (1 : Fin 3))
    (fun _ ↦ by simpa using sliceMatrix_rowOf (matrixMul H E) (2 : Fin 3))
  have hsum := hloop.add hbranch
  obtain ⟨e, he, hc⟩ := hsum
  -- The finish at component 1.
  choose f hf hS using hswitch
  have htarget : (∑ i ∈ Finset.range L, lutBranchFlat sigma hodd μ sp fam i) 1 +
      reducePoly q n (substitution (sigma ⟨L, by omega⟩).toNat (hodd ⟨L, by omega⟩) μ) •
        (reducePoly q n (substitution (sigma ⟨L, by omega⟩).toNat (hodd ⟨L, by omega⟩) sp) •
          fam 0 1 ⟨L, by omega⟩ + fam 1 1 ⟨L, by omega⟩) =
      reducePoly q n (Δ' * ν) • gq +
        reduceMatrix q n 1 m' (∑ i, substitution (sigma i).toNat (hodd i) μ • f i) := by
    subst hL
    rw [Finset.sum_apply, Finset.sum_range]
    have hsplit : ∑ i : Fin L, lutBranchFlat sigma hodd μ sp fam i 1 +
        reducePoly q n (substitution (sigma ⟨L, by omega⟩).toNat (hodd ⟨L, by omega⟩) μ) •
          (reducePoly q n (substitution (sigma ⟨L, by omega⟩).toNat (hodd ⟨L, by omega⟩) sp) •
            fam 0 1 ⟨L, by omega⟩ + fam 1 1 ⟨L, by omega⟩) =
        ∑ i : Fin (L + 1), reducePoly q n (substitution (sigma i).toNat (hodd i) μ) •
          (reducePoly q n (substitution (sigma i).toNat (hodd i) sp) • fam 0 1 i + fam 1 1 i) := by
      rw [Fin.sum_univ_castSucc]
      congr 1
      refine Finset.sum_congr rfl fun i _ ↦ ?_
      rw [lutBranchFlat, dif_pos (by omega)]
      rfl
    rw [hsplit]
    simp only [hS]
    funext row column
    simp only [Matrix.sum_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, reduceMatrix]
    have hterm : ∀ i ∈ (Finset.univ : Finset (Fin (L + 1))),
        reducePoly q n (substitution (sigma i).toNat (hodd i) μ) *
          (C i * reducePoly q n Δ' * gq row column + reducePoly q n (f i row column)) =
        (reducePoly q n (substitution (sigma i).toNat (hodd i) μ) * C i) *
            (reducePoly q n Δ' * gq row column) +
          reducePoly q n (substitution (sigma i).toNat (hodd i) μ * f i row column) := by
      intro i _
      simp only [map_mul]
      ring
    rw [Finset.sum_congr rfl hterm, Finset.sum_add_distrib, ← Finset.sum_mul, hwalsh, map_sum]
    simp only [map_mul, Matrix.sum_apply]
    ring
  refine ⟨e 1 - ∑ i, substitution (sigma i).toNat (hodd i) μ • f i, coeffBound_sub (he 1) ?_, ?_⟩
  · have := coeffBound_sum Finset.univ (fun i _ ↦ coeffBound_smul hn (hμ i) (hf i))
    simpa [Finset.sum_const, Finset.card_univ, Fintype.card_fin] using this
  · have h1 := hc 1
    simp only [Pi.add_apply] at h1
    show Rl 1 + (Y + X) = _
    rw [h1, htarget]
    simp only [reduceMatrix_sub, Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one]
    abel

set_option maxHeartbeats 64000000 in
/-- **A LUT round at one modulus, read at component `j`.** The generated loop's first `L`
branches (`hloop`, both components) and the last branch's component `1` only (the graph computes
nothing else of it when only that component is read): the summed row `R_l 1 + (Y + X)` is
`s̄ K₀ + K₁ - Δ' ν g_q` plus an error, with `K_k = K_l k 1 + row (k + 1) of H E`. -/
theorem lut_round_flat_at (hn : 0 < n) (jc : Fin 2) {log : Nat} (hpow : n = 2 ^ log)
    {q m m' bound W L : Nat} (hL : L + 1 = W) (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    (sigma : Fin W → Int) (hodd : ∀ i, Odd (sigma i).toNat)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {g' : ExactMatrix q n 1 m} {g : ExactMatrix q n 1 m}
    (hg' : gadgetMatrixRuns backend base digits g') (hg : centeredRebaseRuns g' g)
    {c : Fin 2 → ExactMatrix q n 1 m} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (hinput : Rel sm A (fun j ↦ reducePoly q n (((1 : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {fam : Fin 2 → Fin 2 → Fin W → ExactMatrix q n 1 m'}
    {Kl : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {Rl : Fin 2 → ExactMatrix q n 1 m'} {Bl : Nat}
    (hloop : Rel sm Kl (∑ i ∈ Finset.range L, lutBranchFlat sigma hodd μ sp fam i) Rl Bl)
    {h3 : Fin 2 → ExactMatrix q n 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {Bh : Nat}
    (hhelp : Rel sm Ad (fun j ↦ reducePoly q n (((1 : Nat) : Int) *
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
    {Xb : ExactMatrix q n (1 + 1) m'}
    (hXb : concatRows (fam 0 jc ⟨L, by omega⟩) (fam 1 jc ⟨L, by omega⟩) Xb)
    {D' : ExactMatrix q n (m + m) m'} (hD' : gadgetDecomposeRuns backend base digits Xb D')
    {D : ExactMatrix q n (m + m) m'} (hD : compactCenteredRebaseRuns bound D' D)
    {M0 M1 : ExactMatrix q n 1 m'}
    (hM0 : sliceMatrix (matrixMul both D) 1 2 0 m' M0)
    (hM1 : sliceMatrix (matrixMul both D) 2 3 0 m' M1)
    {Z : ExactMatrix q n (1 + 1) m'} (hZ : concatRows M0 M1 Z)
    {E' : ExactMatrix q n (m + m) m'} (hE' : gadgetDecomposeRuns backend base digits Z E')
    {E : ExactMatrix q n (m + m) m'} (hE : compactCenteredRebaseRuns bound E' E)
    {hl01' hr01' : ExactMatrix q n (1 + 1) m} {hl' hr' : ExactMatrix q n (1 + 1 + 1) m}
    {H : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hhl01 : concatRows (h3 0) (Ad 0 0) hl01') (hhl : concatRows hl01' (Ad 1 0) hl')
    (hhr01 : concatRows (h3 1) (Ad 0 1) hr01') (hhr : concatRows hr01' (Ad 1 1) hr')
    (hH : concatColumns hl' hr' H)
    {Y X : ExactMatrix q n 1 m'}
    (hY : sliceMatrix (matrixMul H E) 0 1 0 m' Y)
    (hX : sliceMatrix (matrixMul both D) 0 1 0 m' X)
    {Δ' ν : ErrorPoly n} {gq : ExactMatrix q n 1 m'} {C : Fin W → ExactPoly q n} {BB : Nat}
    (hswitch : ∀ i : Fin W, ∃ e : ErrorMatrix n 1 m', CoeffBound e BB ∧
      reducePoly q n (substitution (sigma i).toNat (hodd i) sp) • fam 0 jc i + fam 1 jc i =
        (C i * reducePoly q n (Δ' * unitPair sp jc)) • gq + reduceMatrix q n 1 m' e)
    (hμ : ∀ i, polyNorm (substitution (sigma i).toNat (hodd i) μ) ≤ 1)
    (hwalsh : ∑ i, reducePoly q n (substitution (sigma i).toNat (hodd i) μ) * C i =
      reducePoly q n ν) :
    ∃ e : ErrorMatrix n 1 m', CoeffBound e
        (Bl + ((m + m) * n * Bh * bound + (m + m) * n * B * bound) + W * (n * 1 * BB)) ∧
      matrixAdd (Rl jc) (matrixAdd Y X) =
        reducePoly q n sm • (Kl 0 jc + rowOf (matrixMul H E) 1) +
          (Kl 1 jc + rowOf (matrixMul H E) 2) - reducePoly q n (Δ' * ν * unitPair sp jc) • gq +
          reduceMatrix q n 1 m' e := by
  -- The last branch, its component-1 data duplicated to both components.
  have hbranch := lut_branch_flat (index := sigma ⟨L, by omega⟩) (log := log) (bound := bound)
    hn hpow hq1 layout hlayout hexact hg' hg hsm hinput hr0 hr1 hr2 hr3 hr4 hr5 hl01 hl hr01 hr
    hboth (bsw := fun k _ ↦ fam k jc ⟨L, by omega⟩) (Xb := fun _ ↦ Xb) (fun _ ↦ hXb)
    (D' := fun _ ↦ D') (fun _ ↦ hD') (D := fun _ ↦ D) (fun _ ↦ hD)
    (M := fun k _ ↦ ![M0, M1] k) (fun _ ↦ hM0) (fun _ ↦ hM1)
    (Z := fun _ ↦ Z) (fun _ ↦ hZ) (E' := fun _ ↦ E') (fun _ ↦ hE') (E := fun _ ↦ E) (fun _ ↦ hE)
    (hodd _) hhelp hhl01 hhl hhr01 hhr hH
    (Y := fun _ ↦ Y) (X := fun _ ↦ X)
    (K := fun k _ ↦ ![rowOf (matrixMul H E) 1, rowOf (matrixMul H E) 2] k) (fun _ ↦ hY) (fun _ ↦ hX)
    (fun _ ↦ by simpa using sliceMatrix_rowOf (matrixMul H E) (1 : Fin 3))
    (fun _ ↦ by simpa using sliceMatrix_rowOf (matrixMul H E) (2 : Fin 3))
  have hsum := hloop.add hbranch
  obtain ⟨e, he, hc⟩ := hsum
  -- The finish at component 1.
  choose f hf hS using hswitch
  have htarget : (∑ i ∈ Finset.range L, lutBranchFlat sigma hodd μ sp fam i) jc +
      reducePoly q n (substitution (sigma ⟨L, by omega⟩).toNat (hodd ⟨L, by omega⟩) μ) •
        (reducePoly q n (substitution (sigma ⟨L, by omega⟩).toNat (hodd ⟨L, by omega⟩) sp) •
          fam 0 jc ⟨L, by omega⟩ + fam 1 jc ⟨L, by omega⟩) =
      reducePoly q n (Δ' * ν * unitPair sp jc) • gq +
        reduceMatrix q n 1 m' (∑ i, substitution (sigma i).toNat (hodd i) μ • f i) := by
    subst hL
    rw [Finset.sum_apply, Finset.sum_range]
    have hsplit : ∑ i : Fin L, lutBranchFlat sigma hodd μ sp fam i jc +
        reducePoly q n (substitution (sigma ⟨L, by omega⟩).toNat (hodd ⟨L, by omega⟩) μ) •
          (reducePoly q n (substitution (sigma ⟨L, by omega⟩).toNat (hodd ⟨L, by omega⟩) sp) •
            fam 0 jc ⟨L, by omega⟩ + fam 1 jc ⟨L, by omega⟩) =
        ∑ i : Fin (L + 1), reducePoly q n (substitution (sigma i).toNat (hodd i) μ) •
          (reducePoly q n (substitution (sigma i).toNat (hodd i) sp) • fam 0 jc i + fam 1 jc i) := by
      rw [Fin.sum_univ_castSucc]
      congr 1
      refine Finset.sum_congr rfl fun i _ ↦ ?_
      rw [lutBranchFlat, dif_pos (by omega)]
      rfl
    rw [hsplit]
    simp only [hS]
    funext row column
    simp only [Matrix.sum_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, reduceMatrix]
    have hterm : ∀ i ∈ (Finset.univ : Finset (Fin (L + 1))),
        reducePoly q n (substitution (sigma i).toNat (hodd i) μ) *
          (C i * reducePoly q n (Δ' * unitPair sp jc) * gq row column + reducePoly q n (f i row column)) =
        (reducePoly q n (substitution (sigma i).toNat (hodd i) μ) * C i) *
            (reducePoly q n (Δ' * unitPair sp jc) * gq row column) +
          reducePoly q n (substitution (sigma i).toNat (hodd i) μ * f i row column) := by
      intro i _
      simp only [map_mul]
      ring
    rw [Finset.sum_congr rfl hterm, Finset.sum_add_distrib, ← Finset.sum_mul, hwalsh, map_sum]
    simp only [map_mul, Matrix.sum_apply]
    ring
  refine ⟨e jc - ∑ i, substitution (sigma i).toNat (hodd i) μ • f i, coeffBound_sub (he jc) ?_, ?_⟩
  · have := coeffBound_sum Finset.univ (fun i _ ↦ coeffBound_smul hn (hμ i) (hf i))
    simpa [Finset.sum_const, Finset.card_univ, Fintype.card_fin] using this
  · have h1 := hc jc
    simp only [Pi.add_apply] at h1
    show Rl jc + (Y + X) = _
    rw [h1, htarget]
    simp only [reduceMatrix_sub, Fin.isValue, Matrix.cons_val_zero, Matrix.cons_val_one]
    abel

end Dio2Proof
