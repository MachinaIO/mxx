import ExponentStacking

/-!
# One LUT branch

A LUT evaluates one branch per automorphism `σ`: the input encoding is routed by `σ`, switched by
the decomposition `D_σ` of the preprocessed ciphertext `B_σ`, moved down one level, and masked by
the encrypted helper acting through the decomposition of the switched key. `lut_branch` states the
result of one branch for the generated computation itself.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime
open scoped BigOperators

variable {n : Nat}

/-! ## Gadgets have constant entries -/

private theorem castExactMatrix_constant {a b r c : Nat} (h : a = b) (M : ExactMatrix a n r c)
    (hM : ∀ i j, ∃ x : ZMod a, M i j = algebraMap (ZMod a) (ExactPoly a n) x) :
    ∀ i j, ∃ x : ZMod b, castExactMatrix h M i j = algebraMap (ZMod b) (ExactPoly b n) x := by
  subst h
  exact hM

/-- Every entry of a runtime gadget matrix is a constant polynomial. -/
theorem gadgetMatrixRuns_constant {q m : Nat} {backend : BackendContext} {base digits : Int}
    {g : ExactMatrix q n 1 m} (hg : gadgetMatrixRuns backend base digits g) :
    ∀ i j, ∃ x : ZMod q, g i j = algebraMap (ZMod q) (ExactPoly q n) x := by
  obtain ⟨layout, _, _, _, hw, hg⟩ := hg
  intro i j
  rw [hg, castMatrixColumns_apply]
  unfold regularGadgetMatrix fullGadgetMatrix
  apply castExactMatrix_constant
  intro row column
  simp only [regularGadgetUnflattened]
  split
  · exact ⟨_, rfl⟩
  · exact ⟨0, by simp⟩

private theorem coeff_algebraMap {R : Type} [CommRing R] (hn : 0 < n) (x : R) (i : Fin n) :
    (algebraMap R (Negacyclic n R) x).coeff i = if i.val = 0 then x else 0 := by
  by_cases hR : Nontrivial R
  · have h := Negacyclic.coeff_smul x (1 : Negacyclic n R) i
    rw [mul_one] at h
    rw [h, show (1 : Negacyclic n R) =
      AdjoinRoot.root (negacyclicModulus n R) ^ ((⟨0, hn⟩ : Fin n) : Nat) by simp,
      Negacyclic.coeff_root_pow hn]
    split <;> split <;> simp_all [Fin.ext_iff, eq_comm]
  · have : Subsingleton R := not_nontrivial_iff_subsingleton.mp hR
    exact Subsingleton.elim _ _

/-- The centered lift of a constant polynomial is an integer constant. -/
theorem centeredIntLift_algebraMap {q : Nat} (hn : 0 < n) (x : ZMod q) :
    ∃ c : Int, centeredIntLift (algebraMap (ZMod q) (ExactPoly q n) x) = (c : ErrorPoly n) := by
  refine ⟨if x.val ≤ q / 2 then (x.val : Int) else (x.val : Int) - q, negacyclic_ext hn fun i ↦ ?_⟩
  rw [centeredIntLift, intPoly_coeff hn, coeff_algebraMap hn, intCast_eq_algebraMap,
    coeff_algebraMap hn]
  split
  · simp [Int.ofNat_eq_natCast]
  · simp [ZMod.val_zero]

/-- A gadget re-encoded at another modulus is fixed by every substitution. -/
theorem substituteMatrix_rebasedGadget {q q' m : Nat} (hn : 0 < n) {backend : BackendContext}
    {base digits : Int} {g' : ExactMatrix q' n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g) {index : Nat} (hodd : Odd index) :
    substituteMatrix index hodd g = g := by
  rw [centeredRebaseRuns_reduce hg]
  funext i j
  obtain ⟨x, hx⟩ := gadgetMatrixRuns_constant hg' i j
  obtain ⟨c, hc⟩ := centeredIntLift_algebraMap (n := n) hn x
  simp only [substituteMatrix, hx, hc, map_intCast]

/-- `blockModSwitchRuns_scaleUp` with the dropped factor named. -/
theorem blockModSwitchRuns_scaleUp' {q q' p rows columns : Nat} (hn : 0 < n) (hq : q = p * q')
    {basis : List Nat} {input : ExactMatrix q n rows columns} {output : ExactMatrix q' n rows columns}
    (h : blockModSwitchRuns basis 1 input output) :
    ∃ remainder : ErrorMatrix n rows columns, CoeffBound remainder (p / 2) ∧
      scaleUpMatrix q p output = input - reduceMatrix q n rows columns remainder := by
  have hpq : q / q' = p := by
    obtain ⟨_, _, _, _, hq'1, _⟩ := h
    rw [hq, Nat.mul_div_cancel _ (by omega)]
  have := blockModSwitchRuns_scaleUp hn h
  rwa [hpq] at this

/-! ## Routed targets -/

theorem substituteMatrix_smul {q rows columns : Nat} {index : Nat} (hodd : Odd index)
    (x : ErrorPoly n) (g : ExactMatrix q n rows columns) :
    substituteMatrix index hodd (reducePoly q n x • g) =
      reducePoly q n (substitution index hodd x) • substituteMatrix index hodd g := by
  funext i j
  simp [substituteMatrix, reducePoly_substitution]

theorem substitution_unitPair {R : Type} [CommRing R] {index : Nat} (hodd : Odd index)
    (x : Negacyclic n R) (j : Fin 2) :
    substitution index hodd (unitPair x j) = unitPair (substitution index hodd x) j := by
  fin_cases j <;> simp [unitPair]

/-- The routed encoding target: `σ((p μ r_j) g) = (p σ(μ) σ(r_j)) g` for a re-encoded gadget. -/
theorem substituteMatrix_target {q q' m : Nat} (hn : 0 < n) {backend : BackendContext}
    {base digits : Int} {g' : ExactMatrix q' n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g) {index : Nat} (hodd : Odd index)
    (p : Nat) (μ x : ErrorPoly n) :
    substituteMatrix index hodd (reducePoly q n ((p : Int) * μ * x) • g) =
      reducePoly q n ((p : Int) * substitution index hodd μ * substitution index hodd x) • g := by
  rw [substituteMatrix_smul, substituteMatrix_rebasedGadget hn hg' hg]
  simp

/-- `![x 0, x 1] = x`. -/
theorem pair_eta {α : Type} (x : Fin 2 → α) : ![x 0, x 1] = x := by
  funext j
  fin_cases j <;> rfl

/-! ## The branch -/

set_option maxHeartbeats 1000000 in
/-- One LUT branch computed as the generated graph computes it. The routed input `σ(c)` is
switched by the decomposition `D_j` of the columns of the preprocessed `B`, the switched key is
moved down (`Asw`), and the helper `h` encrypted for `σ` acts through the decomposition `E_j` of
`Asw`. The branch output `Y + X` is related under the original mask to the helper key `K`, with
the target `p σ(μ) (σ(r₀) B_{0j} + B_{1j})` lifted from the next modulus. -/
theorem lut_branch {q q' p m m' log bound : Nat} (hn : 0 < n) (hpow : n = 2 ^ log)
    (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q')
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    {g' : ExactMatrix q' n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g)
    {sm sp μ : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {c : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly q n ((p : Int) * μ * unitPair sp j) • g) c B)
    {index : Int} {rc0 rA00 rA10 rc1 rA01 rA11 : ExactMatrix q n 1 m}
    (hr0 : ringAutomorphismRuns index (c 0) rc0) (hr1 : ringAutomorphismRuns index (A 0 0) rA00)
    (hr2 : ringAutomorphismRuns index (A 1 0) rA10) (hr3 : ringAutomorphismRuns index (c 1) rc1)
    (hr4 : ringAutomorphismRuns index (A 0 1) rA01) (hr5 : ringAutomorphismRuns index (A 1 1) rA11)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows rc0 rA00 l01) (hl : concatRows l01 rA10 l)
    (hr01 : concatRows rc1 rA01 r01) (hr : concatRows r01 rA11 r) (hboth : concatColumns l r both)
    {bsw : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'} {Xb : Fin 2 → ExactMatrix q' n (1 + 1) m'}
    (hXb : ∀ j, concatRows (bsw 0 j) (bsw 1 j) (Xb j))
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
    {h : Fin 2 → ExactMatrix q n 1 m} {Ad : Fin 2 → Fin 2 → ExactMatrix q n 1 m} {Bh : Nat}
    (hodd : Odd index.toNat)
    (hhelp : Rel sm Ad (fun j ↦ reducePoly q n ((p : Int) *
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
    Rel sm K (fun j ↦ scaleUpMatrix q p (reducePoly q' n (substitution index.toNat hodd μ) •
        (reducePoly q' n (substitution index.toNat hodd sp) • bsw 0 j + bsw 1 j)))
      (fun j ↦ Y j + X j)
      ((m + m) * n * Bh * bound + (m + m) * n * B * bound + (n * 1 * (p / 2) + p / 2)) := by
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
      (reducePoly q n ((p : Int) * μ * unitPair sp i) • g)) (D j) =
      scaleUpMatrix q p (reducePoly q' n (substitution index.toNat hodd μ) •
        (reducePoly q' n (substitution index.toNat hodd sp) • bsw 0 j + bsw 1 j)) := by
    intro j
    simp only [substituteMatrix_target hn hg' hg]
    have hgad := gadget_action_scaleUp hq hq' hn layout hlayout hexact hg' hg (hD' j) (hD j)
      (fun i ↦ substitution index.toNat hodd μ * substitution index.toNat hodd (unitPair sp i))
    simp only [← mul_assoc] at hgad
    rw [hgad, (concatRows_two (hXb j)).1, (concatRows_two (hXb j)).2]
    congr 1
    simp only [unitPair, Matrix.cons_val_zero, Matrix.cons_val_one, map_one, map_mul, mul_one,
      smul_add, smul_smul]
  -- The helper target after its action.
  have hth : ∀ j, pairMul (fun i ↦ reducePoly q n ((p : Int) *
      unitPair (substitution index.toNat hodd sm) i) • g) (E j) =
      scaleUpMatrix q p (reducePoly q' n (substitution index.toNat hodd sm) • Asw 0 j + Asw 1 j) := by
    intro j
    rw [gadget_action_scaleUp hq hq' hn layout hlayout hexact hg' hg (hE' j) (hE j),
      (concatRows_two (hZ j)).1, (concatRows_two (hZ j)).2]
    simp [unitPair]
  -- ModDown of the switched key.
  have hpq : q / q' = p := by rw [hq, Nat.mul_div_cancel _ (by omega)]
  have hsw : ∀ k j, ∃ r : ErrorMatrix n 1 m', CoeffBound r (p / 2) ∧
      scaleUpMatrix q p (Asw k j) = pairMul (fun i ↦ substituteMatrix index.toNat hodd (A k i)) (D j) -
        reduceMatrix q n 1 m' r := by
    intro k j
    obtain ⟨r, hr, hrel⟩ := blockModSwitchRuns_scaleUp hn (hAsw k j)
    rw [hpq] at hr hrel
    exact ⟨r, hr, by rw [hrel, hMe k j]⟩
  have hsmσ : polyNorm (substitution index.toNat hodd sm) ≤ 1 :=
    (polyNorm_substitution_le hn hodd hcop sm).trans hsm
  have hbranch := Rel.lutBranch hn hq hq' hsmσ hact hsw hhact hth
  simp only [htc] at hbranch
  have hYX : (fun j ↦ Y j + X j) = (fun j ↦ pairMul h (E j)) +
      fun j ↦ pairMul (fun i ↦ substituteMatrix index.toNat hodd (c i)) (D j) := by
    funext j
    rw [Pi.add_apply, hYe j, hXe j]
  have hKfun : K = fun k j ↦ pairMul (Ad k) (E j) := by
    funext k j
    exact hKe k j
  rw [hYX, hKfun]
  exact hbranch.mono (le_of_eq (by ring))

/-! ## Finishing a LUT -/

set_option maxHeartbeats 1000000 in
/-- A LUT moved down one level. The summed branches are related with the target
`Σ_i scaleUp (σ_i(μ) (σ_i(r₀) B^i_{0j} + B^i_{1j}))`, where the switching ciphertexts satisfy
`σ_i(r₀) B^i_{0j} + B^i_{1j} = (C_i Δ' r_j) g' + e^i_j`; when `Σ_i σ_i(μ) C_i = ν`, the result is
an encoding of `ν` at the next level with scale `Δ'` and gadget `g'`. -/
theorem lut_finish {q q' p m' W : Nat} (hn : 0 < n) (hq : q = p * q') (hq1 : 1 < q)
    (hq' : 1 < q') (hp : 0 < p) {sm sp μ ν Δ' : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    (σ : Fin W → ErrorPoly n → ErrorPoly n) (hσμ : ∀ i, polyNorm (σ i μ) ≤ 1)
    {bsw : Fin W → Fin 2 → Fin 2 → ExactMatrix q' n 1 m'}
    {Ksum : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {Rsum : Fin 2 → ExactMatrix q n 1 m'} {Bs : Nat}
    (hsum : Rel sm Ksum (fun j ↦ ∑ i, scaleUpMatrix q p (reducePoly q' n (σ i μ) •
      (reducePoly q' n (σ i sp) • bsw i 0 j + bsw i 1 j))) Rsum Bs)
    {g' : ExactMatrix q' n 1 m'} {C : Fin W → ExactPoly q' n} {eB : Fin W → Fin 2 → ErrorMatrix n 1 m'}
    {BB : Nat} (hswitch : ∀ i j, reducePoly q' n (σ i sp) • bsw i 0 j + bsw i 1 j =
      (C i * reducePoly q' n (Δ' * unitPair sp j)) • g' + reduceMatrix q' n 1 m' (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (hwalsh : ∑ i, reducePoly q' n (σ i μ) * C i = reducePoly q' n ν) {basis : List Nat}
    {K' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'} {R' : Fin 2 → ExactMatrix q' n 1 m'}
    (hK' : ∀ k j, blockModSwitchRuns basis 1 (Ksum k j) (K' k j))
    (hR' : ∀ j, blockModSwitchRuns basis 1 (Rsum j) (R' j)) :
    Rel sm K' (fun j ↦ reducePoly q' n (Δ' * ν * unitPair sp j) • g') R'
      ((Bs + (n + 2) * (p / 2)) / p + W * (n * 1 * BB)) := by
  -- The summed target.
  let f : Fin 2 → ErrorMatrix n 1 m' := fun j ↦ ∑ i, σ i μ • eB i j
  have htarget : ∀ j, (∑ i, scaleUpMatrix q p (reducePoly q' n (σ i μ) •
      (reducePoly q' n (σ i sp) • bsw i 0 j + bsw i 1 j))) =
      scaleUpMatrix q p (reducePoly q' n (Δ' * ν * unitPair sp j) • g' + reduceMatrix q' n 1 m' (f j)) := by
    intro j
    simp only [hswitch]
    funext row column
    simp only [scaleUpMatrix, Matrix.sum_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul,
      reduceMatrix, f]
    rw [← scaleUp_sum hq hq' hn]
    congr 1
    have hterm : ∀ i ∈ (Finset.univ : Finset (Fin W)), reducePoly q' n (σ i μ) *
        (C i * reducePoly q' n (Δ' * unitPair sp j) * g' row column +
          reducePoly q' n (eB i j row column)) =
        (reducePoly q' n (σ i μ) * C i) * (reducePoly q' n (Δ' * unitPair sp j) * g' row column) +
          reducePoly q' n (σ i μ * eB i j row column) := by
      intro i _
      simp only [map_mul]
      ring
    rw [Finset.sum_congr rfl hterm, Finset.sum_add_distrib, ← Finset.sum_mul, hwalsh,
      map_sum]
    simp only [map_mul]
    ring
  have hsum' : Rel sm Ksum (fun j ↦ scaleUpMatrix q p
      (reducePoly q' n (Δ' * ν * unitPair sp j) • g' + reduceMatrix q' n 1 m' (f j))) Rsum Bs := by
    obtain ⟨e, he, hc⟩ := hsum
    exact ⟨e, he, fun j ↦ by rw [hc j]; simp only []; rw [htarget j]⟩
  have hdown := hsum'.modDown hn hq hq1 hq' hp hsm (fun j ↦ rfl)
    (fun k j ↦ blockModSwitchRuns_scaleUp' hn hq (hK' k j))
    (fun j ↦ blockModSwitchRuns_scaleUp' hn hq (hR' j))
  have hf : ∀ j, CoeffBound (f j) (W * (n * 1 * BB)) := by
    intro j
    have := coeffBound_sum Finset.univ (fun i _ ↦ coeffBound_smul hn (hσμ i) (heB i j))
    simpa [Finset.sum_const, Finset.card_univ, Fintype.card_fin] using this
  exact hdown.absorb hf

/-! ## Private multiplication -/

theorem scaleUpMatrix_one {q rows columns : Nat} (hq : 1 < q) (hn : 0 < n)
    (x : ExactMatrix q n rows columns) : scaleUpMatrix q 1 x = x := by
  funext i j
  simp [scaleUpMatrix, scaleUp, reducePoly_canonicalLift hq hn]

set_option maxHeartbeats 1000000 in
/-- Private multiplication at a level that keeps its modulus: the stacked encoding is acted on by
the decomposition of the columns of the preprocessed RHS `(top, bot)`, whose payload relation
`r₀ top_j + bot_j = T_j + e_B` multiplies the target into `μ T_j`. -/
theorem mul_private {q m m' bound : Nat} (hn : 0 < n) (hq1 : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {g' : ExactMatrix q n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g)
    {sm sp μ : ErrorPoly n} {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m}
    {c : Fin 2 → ExactMatrix q n 1 m} {B : Nat}
    (henc : Rel sm A (fun j ↦ reducePoly q n (((1 : Nat) : Int) * μ * unitPair sp j) • g) c B)
    {l01 r01 : ExactMatrix q n (1 + 1) m} {l r : ExactMatrix q n (1 + 1 + 1) m}
    {both : ExactMatrix q n (1 + 1 + 1) (m + m)}
    (hl01 : concatRows (c 0) (A 0 0) l01) (hl : concatRows l01 (A 1 0) l)
    (hr01 : concatRows (c 1) (A 0 1) r01) (hr : concatRows r01 (A 1 1) r)
    (hboth : concatColumns l r both)
    {top bot : Fin 2 → ExactMatrix q n 1 m'} {Xb : Fin 2 → ExactMatrix q n (1 + 1) m'}
    (hXb : ∀ j, concatRows (top j) (bot j) (Xb j))
    {D' : Fin 2 → ExactMatrix q n (m + m) m'}
    (hD' : ∀ j, gadgetDecomposeRuns backend base digits (Xb j) (D' j))
    {D : Fin 2 → ExactMatrix q n (m + m) m'} (hD : ∀ j, compactCenteredRebaseRuns bound (D' j) (D j))
    {T : Fin 2 → ExactMatrix q n 1 m'} {eB : Fin 2 → ErrorMatrix n 1 m'} {BB : Nat}
    (hrhs : ∀ j, reducePoly q n sp • top j + bot j = T j + reduceMatrix q n 1 m' (eB j))
    (heB : ∀ j, CoeffBound (eB j) BB) (hμ : polyNorm μ ≤ 1)
    {c' : Fin 2 → ExactMatrix q n 1 m'} {A' : Fin 2 → Fin 2 → ExactMatrix q n 1 m'}
    (hc' : ∀ j, sliceMatrix (matrixMul both (D j)) 0 1 0 m' (c' j))
    (hA0 : ∀ j, sliceMatrix (matrixMul both (D j)) 1 2 0 m' (A' 0 j))
    (hA1 : ∀ j, sliceMatrix (matrixMul both (D j)) 2 3 0 m' (A' 1 j)) :
    Rel sm A' (fun j ↦ reducePoly q n μ • T j) c' ((m + m) * n * B * bound + n * 1 * BB) := by
  have hrows := fun j ↦ stacked_row hl01 hl hr01 hr hboth (D j)
  choose d hd using fun j ↦ (hD j).2.2.2
  have hDd : ∀ j, D j = reduceMatrix q n (m + m) m' (d j) := fun j ↦ (hd j).1
  have hact := henc.action hn d (fun j ↦ (hd j).2)
  simp only [← hDd] at hact
  have hce : c' = fun j ↦ pairMul c (D j) := by
    funext j
    rw [sliceMatrix_row (rows := 1 + 1 + 1) 0 (hc' j), ← pair_eta c]
    exact (hrows j).1
  have hAe : A' = fun k j ↦ pairMul (A k) (D j) := by
    funext k j
    match k with
    | 0 =>
      rw [sliceMatrix_row (rows := 1 + 1 + 1) 1 (hA0 j), ← pair_eta (A 0)]
      exact (hrows j).2.1
    | 1 =>
      rw [sliceMatrix_row (rows := 1 + 1 + 1) 2 (hA1 j), ← pair_eta (A 1)]
      exact (hrows j).2.2
  have hq : q = 1 * q := (one_mul q).symm
  have ht : ∀ j, pairMul (fun i ↦ reducePoly q n (((1 : Nat) : Int) * μ * unitPair sp i) • g) (D j) =
      reducePoly q n μ • T j + reduceMatrix q n 1 m' (μ • eB j) := by
    intro j
    have hgad := gadget_action_scaleUp hq hq1 hn layout hlayout hexact hg' hg (hD' j) (hD j)
      (fun i ↦ μ * unitPair sp i)
    simp only [← mul_assoc] at hgad
    rw [hgad, scaleUpMatrix_one hq1 hn, (concatRows_two (hXb j)).1, (concatRows_two (hXb j)).2,
      reduceMatrix_smul, ← smul_add, ← hrhs j]
    simp only [unitPair, Matrix.cons_val_zero, Matrix.cons_val_one, map_mul, mul_one, smul_add,
      smul_smul]
  simp only [ht] at hact
  rw [hce, hAe]
  have habs := hact.absorb (f := fun j ↦ μ • eB j) (F := n * 1 * BB)
    (fun j ↦ coeffBound_smul hn hμ (heB j))
  exact habs

end MxxExponentLut
