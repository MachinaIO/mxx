import TopLut

/-!
# Moving a sum of LUT branches down

A PRF sampler sums, over its members, every branch of the member's sampling LUT placed at `X^o`
(and the branches of the binary offset), and moves the sum down once. `lut_finish_sum` is
`lut_finish` over any finite family of branches, each with its own multiplier `a_i` (the branch's
automorphism of the member's state, times the placement) and switching key `b_i` (the
automorphism of the secret).
-/

namespace RefreshTopProof

open Mxx.Primitives MxxRuntime MxxExponentLut

set_option maxHeartbeats 1000000 in
/-- `lut_finish` over a finite family `ι` of branches with multipliers `a` and secret
automorphisms `b`. -/
theorem lut_finish_sum {ι : Type} [Fintype ι] [DecidableEq ι] {q q' p m' : Nat} (hq : q = p * q')
    (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p) {sm sp ν Δ' : ErrorPoly N} (hsm : polyNorm sm ≤ 1)
    (a b : ι → ErrorPoly N) (ha : ∀ i, polyNorm (a i) ≤ 1)
    {bsw : ι → Fin 2 → Fin 2 → ExactMatrix q' N 1 m'}
    {Ksum : Fin 2 → Fin 2 → ExactMatrix q N 1 m'} {Rsum : Fin 2 → ExactMatrix q N 1 m'} {Bs : Nat}
    (hsum : Rel sm Ksum (fun j ↦ ∑ i, scaleUpMatrix q p (reducePoly q' N (a i) •
      (reducePoly q' N (b i) • bsw i 0 j + bsw i 1 j))) Rsum Bs)
    {g' : ExactMatrix q' N 1 m'} {C : ι → ExactPoly q' N} {eB : ι → Fin 2 → ErrorMatrix N 1 m'}
    {BB : Nat} (hswitch : ∀ i j, reducePoly q' N (b i) • bsw i 0 j + bsw i 1 j =
      (C i * reducePoly q' N (Δ' * unitPair sp j)) • g' + reduceMatrix q' N 1 m' (eB i j))
    (heB : ∀ i j, CoeffBound (eB i j) BB)
    (hwalsh : ∑ i, reducePoly q' N (a i) * C i = reducePoly q' N ν) {basis : List Nat}
    {K' : Fin 2 → Fin 2 → ExactMatrix q' N 1 m'} {R' : Fin 2 → ExactMatrix q' N 1 m'}
    (hK' : ∀ k j, blockModSwitchRuns basis 1 (Ksum k j) (K' k j))
    (hR' : ∀ j, blockModSwitchRuns basis 1 (Rsum j) (R' j)) :
    Rel sm K' (fun j ↦ reducePoly q' N (Δ' * ν * unitPair sp j) • g') R'
      ((Bs + (N + 2) * (p / 2)) / p + Fintype.card ι * (N * 1 * BB)) := by
  let f : Fin 2 → ErrorMatrix N 1 m' := fun j ↦ ∑ i, a i • eB i j
  have htarget : ∀ j, (∑ i, scaleUpMatrix q p (reducePoly q' N (a i) •
      (reducePoly q' N (b i) • bsw i 0 j + bsw i 1 j))) =
      scaleUpMatrix q p (reducePoly q' N (Δ' * ν * unitPair sp j) • g' +
        reduceMatrix q' N 1 m' (f j)) := by
    intro j
    simp only [hswitch]
    funext row column
    simp only [scaleUpMatrix, Matrix.sum_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul,
      reduceMatrix, f]
    rw [← scaleUp_sum hq hq' hN]
    congr 1
    have hterm : ∀ i ∈ (Finset.univ : Finset ι), reducePoly q' N (a i) *
        (C i * reducePoly q' N (Δ' * unitPair sp j) * g' row column +
          reducePoly q' N (eB i j row column)) =
        (reducePoly q' N (a i) * C i) * (reducePoly q' N (Δ' * unitPair sp j) * g' row column) +
          reducePoly q' N (a i * eB i j row column) := by
      intro i _
      simp only [map_mul]
      ring
    rw [Finset.sum_congr rfl hterm, Finset.sum_add_distrib, ← Finset.sum_mul, hwalsh,
      map_sum]
    simp only [map_mul, Matrix.sum_apply, Matrix.smul_apply, smul_eq_mul]
    ring
  have hsum' : Rel sm Ksum (fun j ↦ scaleUpMatrix q p
      (reducePoly q' N (Δ' * ν * unitPair sp j) • g' + reduceMatrix q' N 1 m' (f j))) Rsum Bs := by
    obtain ⟨e, he, hc⟩ := hsum
    exact ⟨e, he, fun j ↦ by rw [hc j]; simp only []; rw [htarget j]⟩
  have hdown := hsum'.modDown hN hq hq1 hq' hp hsm (fun j ↦ rfl)
    (fun k j ↦ blockModSwitchRuns_scaleUp' hN hq (hK' k j))
    (fun j ↦ blockModSwitchRuns_scaleUp' hN hq (hR' j))
  have hf : ∀ j, CoeffBound (f j) (Fintype.card ι * (N * 1 * BB)) := by
    intro j
    have := coeffBound_sum Finset.univ (fun i _ ↦ coeffBound_smul hN (ha i) (heB i j))
    simpa [Finset.sum_const, Finset.card_univ] using this
  exact hdown.absorb hf

/-- The block modulus switch as a function. -/
noncomputable def blockModSwitch {q p rows columns : Nat} (basis : List Nat) (plaintext : Int)
    (input : ExactMatrix q N rows columns) : ExactMatrix p N rows columns :=
  fun row column ↦ polynomialOfCoefficients (fun i ↦
    let auxiliary := (basis.filter (fun prime ↦ p % prime != 0)).prod
    let z := Int.ofNat ((input row column).coeff i).val
    let correction := rnsCentered auxiliary (z * rnsInverse auxiliary plaintext)
    ((z - plaintext * correction) / Int.ofNat auxiliary) % Int.ofNat p)

/-- A block modulus switch that runs on one input runs on every input of the same moduli. -/
theorem blockModSwitchRuns_fn {q p rows columns rows' columns' : Nat} {basis : List Nat}
    {plaintext : Int} {x : ExactMatrix q N rows columns} {y : ExactMatrix p N rows columns}
    (h : blockModSwitchRuns basis plaintext x y) (x' : ExactMatrix q N rows' columns') :
    blockModSwitchRuns basis plaintext x' (blockModSwitch basis plaintext x' : ExactMatrix p N _ _) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, -⟩ := h
  exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, fun _ _ ↦ rfl⟩

theorem blockModSwitchRuns_eq {q p rows columns : Nat} {basis : List Nat} {plaintext : Int}
    {x : ExactMatrix q N rows columns} {y : ExactMatrix p N rows columns}
    (h : blockModSwitchRuns basis plaintext x y) : y = blockModSwitch basis plaintext x :=
  funext fun r ↦ funext fun c ↦ by
    rw [h.2.2.2.2.2.2.2.2.2.2.2 r c, (blockModSwitchRuns_fn h x).2.2.2.2.2.2.2.2.2.2.2 r c]

theorem smul_scaleUpMatrix {q q' p rows columns : Nat} (hq : q = p * q') (hq' : 1 < q')
    (a : ErrorPoly N) (x : ExactMatrix q' N rows columns) :
    reducePoly q N a • scaleUpMatrix q p x = scaleUpMatrix q p (reducePoly q' N a • x) := by
  funext r c
  simp only [scaleUpMatrix, Matrix.smul_apply, smul_eq_mul, scaleUp_reducePoly_mul hq hq' hN]

theorem smul_sum_matrix {q rows columns : Nat} {ι : Type} (s : Finset ι) (a : ExactPoly q N)
    (x : ι → ExactMatrix q N rows columns) : a • ∑ i ∈ s, x i = ∑ i ∈ s, a • x i :=
  Finset.smul_sum

end RefreshTopProof
