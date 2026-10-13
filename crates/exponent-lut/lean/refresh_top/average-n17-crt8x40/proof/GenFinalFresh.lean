import GenFinalLink
import GenRelLut
import GenInstLemmas

/-!
# The fresh-noise row of the online final combination, for any parameter set

The online instance builds the fresh row from the PRF sampler members: member `f` (a `P`-branch
sampling LUT on `μ_f`, placed at `X^(e_f)`) is summed per column (`Step::Sum`, `C` columns of `W`
members), the second component of the column row is moved down to the instance ring `R`, and each
column block is multiplied by the compact digits of `Q_q` and assembled (`sum.columns.slice-step`).

* `polyNorm_Xe_mul_le`, `blockModSwitchRuns_blk`.
* `fresh_block_down`: a column of placed sampler members, moved down: target
  `Δ' (Σ_j X^(e_j) ν_j) unitPair sp • g'` (ν the members' LUT values, `hwalsh`).
* `fresh_column_value`: the column's second row times the digits of `Q_q`:
  `maskPart s K - (Δ' ν_c) Q_q + e` (1 × 1).
* `fresh_row_gen`: the assembled row: `maskPart s Kf - Q_q • reduce E + reduce ef` with
  `E 0 c = Δ' Σ_j X^(e_j) ν_(c,j)` — the form `GenFinal.final_gen` needs (`hf`).
-/

namespace RefreshTopProof.GenFinal

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof GenDecode Gen
open GenIface GenInst
open scoped BigOperators

variable {n : Nat}

theorem polyNorm_Xe_mul_le (hn : 1 < n) (e : Nat) (y : ErrorPoly n) :
    polyNorm (Xe (n := n) e * y) ≤ polyNorm y :=
  polyNorm_root_pow_mul_le hn e y

/-- The block modulus switch acts entrywise: it switches every column block. -/
theorem blockModSwitchRuns_blk {q p C m : Nat} {basis : List Nat} {x : ExactMatrix q n 1 (C * m)}
    {y : ExactMatrix p n 1 (C * m)} (h : blockModSwitchRuns basis 1 x y) (c : Fin C) :
    blockModSwitchRuns basis 1 (blk x c) (blk y c) := by
  obtain ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12⟩ := h
  exact ⟨h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, fun r l ↦ h12 r _⟩

set_option maxHeartbeats 16000000 in
/-- **A column of placed sampler members, moved down.** The column's relation has the target
`Σ_j X^(e_j) • Σ_k branch_k(μ_j)` (`P` branches of a LUT on `μ_j`); with the switching keys
(`hswitch`, scale `Δ'`) and the members' LUT values `ν_j` (`hwalsh`), its move-down to `q'` has
target `Δ' (Σ_j X^(e_j) ν_j) unitPair sp • g'`. -/
theorem fresh_block_down (hn : 1 < n) {q q' p m' W P : Nat} (hq : q = p * q') (hq1 : 1 < q)
    (hq' : 1 < q') (hp : 0 < p) {sm sp Δ' : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    (sigma : Fin P → Int) (hodd : ∀ i, Odd (sigma i).toNat)
    (fam : Fin 2 → Fin 2 → Fin P → ExactMatrix q' n 1 m') (e : Fin W → Nat)
    (μ : Fin W → ErrorPoly n)
    (hμ : ∀ j k, polyNorm (substitution (sigma k).toNat (hodd k) (μ j)) ≤ 1)
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {c : Fin 2 → ExactMatrix q n 1 m'} {B : Nat}
    (hcol : Rel sm A (fun j ↦ ∑ jj : Fin W, reducePoly q n (Xe (n := n) (e jj)) •
      ∑ k : Fin P, branchTargetG (q := q) (p := p) sigma hodd (μ jj) sp fam k j) c B)
    {g' : ExactMatrix q' n 1 m'} {C : Fin P → ExactPoly q' n} {eB : Fin P → Fin 2 → ErrorMatrix n 1 m'}
    {BB : Nat}
    (hswitch : ∀ k j, reducePoly q' n (substitution (sigma k).toNat (hodd k) sp) • fam 0 j k + fam 1 j k =
      (C k * reducePoly q' n (Δ' * unitPair sp j)) • g' + reduceMatrix q' n 1 m' (eB k j))
    (heB : ∀ k j, CoeffBound (eB k j) BB)
    (ν : Fin W → ErrorPoly n)
    (hwalsh : ∀ jj, ∑ k, reducePoly q' n (substitution (sigma k).toNat (hodd k) (μ jj)) * C k =
      reducePoly q' n (ν jj))
    {basis : List Nat} {K' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'} {R' : Fin 2 → ExactMatrix q' n 1 m'}
    (hK' : ∀ k j, blockModSwitchRuns basis 1 (A k j) (K' k j))
    (hR' : ∀ j, blockModSwitchRuns basis 1 (c j) (R' j)) :
    Rel sm K' (fun j ↦ reducePoly q' n (Δ' * (∑ jj, Xe (n := n) (e jj) * ν jj) * unitPair sp j) • g') R'
      ((B + (n + 2) * (p / 2)) / p + W * P * (n * 1 * BB)) := by
  have hn0 : 0 < n := by omega
  let a : Fin W × Fin P → ErrorPoly n := fun x ↦ Xe (n := n) (e x.1) * substitution (sigma x.2).toNat (hodd x.2) (μ x.1)
  let b : Fin W × Fin P → ErrorPoly n := fun x ↦ substitution (sigma x.2).toNat (hodd x.2) sp
  have hcol' := hcol.target_eq (t' := fun j ↦ ∑ x : Fin W × Fin P, scaleUpMatrix q p
      (reducePoly q' n (a x) • (reducePoly q' n (b x) • fam 0 j x.2 + fam 1 j x.2))) (by
    funext j
    rw [Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun jj _ ↦ ?_
    rw [Finset.smul_sum]
    refine Finset.sum_congr rfl fun k _ ↦ ?_
    simp only [branchTargetG, a, b]
    rw [smul_scaleUpMatrix_gen hn0 hq hq', smul_smul, ← map_mul])
  have hfin := lut_finish_sum_gen (ι := Fin W × Fin P) hn0 hq hq1 hq' hp hsm a b
    (fun x ↦ (polyNorm_Xe_mul_le hn _ _).trans (hμ x.1 x.2))
    (bsw := fun x k j ↦ fam k j x.2) hcol' (C := fun x ↦ C x.2) (Δ' := Δ')
    (fun x j ↦ hswitch x.2 j) (fun x j ↦ heB x.2 j)
    (ν := ∑ jj, Xe (n := n) (e jj) * ν jj) (by
      rw [Fintype.sum_prod_type, map_sum]
      refine Finset.sum_congr rfl fun jj _ ↦ ?_
      simp only [a, map_mul, mul_assoc, ← Finset.mul_sum]
      rw [hwalsh jj])
    hK' hR'
  refine hfin.mono (le_of_eq ?_)
  simp [Fintype.card_prod, Fintype.card_fin, Nat.mul_assoc]

/-- **The column value.** The second row of a moved-down column (`R' 1`, target `(Δ' ν) • g'`)
times the digits `D` of `Q_q` (`g' D = Q_q`): `maskPart s (K' · 1 D) - (Δ' ν) Q_q + e D`. -/
theorem fresh_column_value (hn : 0 < n) {R m' : Nat} {sm : ErrorPoly n}
    {K' : Fin 2 → Fin 2 → ExactMatrix R n 1 m'}
    {T R' : Fin 2 → ExactMatrix R n 1 m'} {B : Nat} (h : Rel sm K' T R' B)
    {g' : ExactMatrix R n 1 m'} {w : ErrorPoly n} (hT : T 1 = reducePoly R n w • g')
    {Qq : Int} {D : ExactMatrix R n m' 1} (hD : g' * D = matrixPolynomial [Qq])
    {DI : ErrorMatrix n m' 1} {BD : Nat} (hDI : D = reduceMatrix R n m' 1 DI) (hDb : CoeffBound DI BD) :
    ∃ e : ErrorMatrix n 1 1, CoeffBound e (m' * n * B * BD) ∧
      R' 1 * D = maskPart (reducePoly R n sm) (fun k ↦ K' k 1 * D) -
        reducePoly R n w • (matrixPolynomial [Qq] : ExactMatrix R n 1 1) + reduceMatrix R n 1 1 e := by
  obtain ⟨e, he, hc⟩ := h
  refine ⟨e 1 * DI, coeffBound_mul hn (he 1) hDb, ?_⟩
  rw [hc 1, hT, reduceMatrix_mul, ← hDI]
  simp only [maskPart, Matrix.add_mul, Matrix.sub_mul, Matrix.smul_mul, hD]

/-- Column `l` of a row of `C` one-entry blocks. -/
def colOf {C : Nat} (l : Fin (C * 1)) : Fin C := ⟨l.val, by have := l.isLt; omega⟩

theorem entry_blk {q C : Nat} (y : ExactMatrix q n (1 * 1) (C * 1)) (l : Fin (C * 1)) :
    y 0 l = blk y (colOf l) 0 0 := by
  simp only [blk, colOf]
  congr 1
  exact Fin.ext (by simp)

/-- **The fresh row.** Every column block of the moved-down second row `x` has a relation with
target `w_c • g'` (`fresh_block_down`), and block `c` of the assembled row is block `c` of `x`
times the digits `D` of `Q_q` (`slice_column_total`): the row is
`maskPart s Kf - Q_q • reduce (w) + reduce ef`. -/
theorem fresh_row_gen (hn : 0 < n) {R C m' : Nat} {sm : ErrorPoly n} {x : ExactMatrix R n 1 (C * m')}
    {K' : Fin C → Fin 2 → Fin 2 → ExactMatrix R n 1 m'} {T R' : Fin C → Fin 2 → ExactMatrix R n 1 m'}
    {B : Nat} (h : ∀ cc, Rel sm (K' cc) (T cc) (R' cc) B) (hR : ∀ cc, R' cc 1 = blk x cc)
    {g' : ExactMatrix R n 1 m'} {w : Fin C → ErrorPoly n} (hT : ∀ cc, T cc 1 = reducePoly R n (w cc) • g')
    {Qq : Int} {D : ExactMatrix R n m' 1} (hD : g' * D = matrixPolynomial [Qq])
    {DI : ErrorMatrix n m' 1} {BD : Nat} (hDI : D = reduceMatrix R n m' 1 DI) (hDb : CoeffBound DI BD)
    {final : ExactMatrix R n (1 * 1) (C * 1)} (hblk : ∀ cc, blk final cc = matrixMul (blk x cc) D) :
    ∃ (Kf : Fin 2 → ExactMatrix R n 1 (C * 1)) (ef : ErrorMatrix n 1 (C * 1)),
      CoeffBound ef (m' * n * B * BD) ∧
      final = maskPart (reducePoly R n sm) Kf - ((Qq : Int) : ExactPoly R n) •
        reduceMatrix R n 1 (C * 1) (fun _ l ↦ w (colOf l)) + reduceMatrix R n 1 (C * 1) ef ∧
      Kf = fun k _ l ↦ (K' (colOf l) k 1 * D) 0 0 := by
  choose e he hcol using fun cc ↦ fresh_column_value hn (h cc) (hT cc) hD hDI hDb
  refine ⟨fun k _ l ↦ (K' (colOf l) k 1 * D) 0 0, fun _ l ↦ e (colOf l) 0 0, fun _ l i ↦ he _ 0 0 i, ?_, rfl⟩
  funext r l
  have hr : r = 0 := Fin.ext (by have := r.isLt; omega)
  subst hr
  rw [entry_blk final l, hblk, matrixMul, ← hR, hcol]
  simp only [maskPart, Matrix.add_apply, Matrix.sub_apply, Matrix.smul_apply, smul_eq_mul, reduceMatrix,
    matrixPolynomial_single]
  ring


/-! ## The LUT values of the fresh members -/

/-- **A LUT's value on a monomial input** (full period, `n = s 2^log`): when the walsh relation
`Σ_k σ_k(μ) C_k = ν` holds with the Fourier coefficients `C` of the table `T` and `μ ≡ X^(s z)`,
then `ν ≡ (-1)^⌊z / 2^log⌋ T(z mod 2^log)`. -/
theorem lut_value_of_walsh {q inverse s log : Nat} (hn : n = s * 2 ^ log) (hnpos : 0 < n)
    (T : Fin (2 ^ log) → ErrorPoly n)
    (hinverse : (inverse : ExactPoly q n) * ((2 ^ log : Nat) : ExactPoly q n) = 1)
    (index : Fin (2 ^ log) → Nat) (hodd : ∀ i, Odd (index i)) (perm : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * 2 ^ log) = 1 + 2 * 1 * (perm i).val)
    {C : Fin (2 ^ log) → ExactPoly q n}
    (hC : ∀ i, C i = lutFourierCoefficient (2 ^ log) (X q ^ s) (inverse : ExactPoly q n)
      (fun k ↦ reducePoly q n (T k)) (index i))
    {μ ν : ErrorPoly n} {z : Nat} (hμ : reducePoly q n μ = reducePoly q n (Xe (n := n) (s * z)))
    (hwalsh : ∑ i, reducePoly q n (substitution (index i) (hodd i) μ) * C i = reducePoly q n ν) :
    reducePoly q n ν = reducePoly q n ((((-1 : Int) ^ (z / 2 ^ log) : Int) : ErrorPoly n) *
        T ⟨z % 2 ^ log, Nat.mod_lt _ (by positivity)⟩) := by
  rw [← hwalsh, ← lut_gen_value hn hnpos T hinverse index hodd perm hindex z]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  rw [hC i, reducePoly_substitution, reducePoly_substitution, hμ]

/-- **The sampler's member value**: the sampling LUT (table `binTable P g`, `P = 2^log`) on the
readout `X^(s readoutExpG P W' z)` is `samplerVal P W' g z` modulo `q`. -/
theorem sampler_value_gen {q inverse s log P W' : Nat} (hP : P = 2 ^ log) (hP2 : 2 ≤ P) (hW : 0 < W')
    (hn : n = s * 2 ^ log) (hnpos : 0 < n) (g : Nat → Int)
    (hinverse : (inverse : ExactPoly q n) * ((2 ^ log : Nat) : ExactPoly q n) = 1)
    (index : Fin (2 ^ log) → Nat) (hodd : ∀ i, Odd (index i)) (perm : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * 2 ^ log) = 1 + 2 * 1 * (perm i).val)
    {C : Fin (2 ^ log) → ExactPoly q n}
    (hC : ∀ i, C i = lutFourierCoefficient (2 ^ log) (X q ^ s) (inverse : ExactPoly q n)
      (fun k ↦ reducePoly q n (binTable (n := n) (2 ^ log) g k)) (index i))
    {μ ν : ErrorPoly n} {z : Nat}
    (hμ : reducePoly q n μ = reducePoly q n (Xe (n := n) (s * readoutExpG P W' z)))
    (hwalsh : ∑ i, reducePoly q n (substitution (index i) (hodd i) μ) * C i = reducePoly q n ν) :
    reducePoly q n ν = reducePoly q n (((samplerVal P W' g (z : Int) : Int)) : ErrorPoly n) := by
  subst hP
  haveI : NeZero (2 ^ log) := ⟨by positivity⟩
  rw [lut_value_of_walsh hn hnpos _ hinverse index hodd perm hindex hC hμ hwalsh]
  have h := sampled_bin (n := n) (P := 2 ^ log) (W' := W') (by omega) (by
    rcases log with _ | l
    · simp at hP2
    · rw [pow_succ]; simp) hW g z
  unfold sampledG at h
  rw [h, samplerVal, Int.toNat_natCast]


/-- **The readout's value**: the PRF readout LUT (exponent table `X^(s₁ readoutVal)`, width
`2^log = W'`) on the PRF output `X^(s₀ z)` is `X^(s₁ readoutExpG P W' z)` modulo `q`. -/
theorem readout_value_of_walsh {q inverse s0 s1 log P : Nat} (hn0 : n = s0 * 2 ^ log)
    (hn1 : n = s1 * P) (hnpos : 0 < n)
    (hinverse : (inverse : ExactPoly q n) * ((2 ^ log : Nat) : ExactPoly q n) = 1)
    (index : Fin (2 ^ log) → Nat) (hodd : ∀ i, Odd (index i)) (perm : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * 2 ^ log) = 1 + 2 * 1 * (perm i).val)
    {C : Fin (2 ^ log) → ExactPoly q n}
    (hC : ∀ i, C i = lutFourierCoefficient (2 ^ log) (X q ^ s0) (inverse : ExactPoly q n)
      (fun k : Fin (2 ^ log) ↦ reducePoly q n (Xe (n := n) (s1 * readoutVal (P / 2) (2 ^ log) k))) (index i))
    {μ ν : ErrorPoly n} {z : Nat} (hμ : reducePoly q n μ = reducePoly q n (Xe (n := n) (s0 * z)))
    (hwalsh : ∑ i, reducePoly q n (substitution (index i) (hodd i) μ) * C i = reducePoly q n ν) :
    reducePoly q n ν = reducePoly q n (Xe (n := n) (s1 * readoutExpG P (2 ^ log) z)) := by
  rw [lut_value_of_walsh hn0 hnpos (fun k : Fin (2 ^ log) ↦ Xe (n := n) (s1 * readoutVal (P / 2) (2 ^ log) k)) hinverse
    index hodd perm hindex hC hμ hwalsh, ← readout_value_gen hn1 z]


set_option maxHeartbeats 16000000 in
/-- **The fresh form** (`hf` of `final_gen`, up to the member values): the `Step::Sum` row `X` of
placed sampler members (column `c`, member `j`: placement `e c j`, LUT input `μ c j`), its second
component moved down (`h434`), and the slice loop's row `final` (block `c` = block `c` of the
moved-down row times the digits `D` of `Q_q`). With switching keys of scale `1` and the members'
LUT values `ν`, `final = maskPart s Kf - Q_q • reduce E + reduce ef`, `E 0 l = Σ_j X^(e c j) ν c j`. -/
theorem fresh_form_gen (hn : 1 < n) {q R p m' W P C : Nat} (hq : q = p * R) (hq1 : 1 < q)
    (hR : 1 < R) (hp : 0 < p) {sm sp : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    (sigma : Fin P → Int) (hodd : ∀ i, Odd (sigma i).toNat)
    (fam : Fin 2 → Fin 2 → Fin P → ExactMatrix R n 1 m') (e : Fin C → Fin W → Nat)
    (μ : Fin C → Fin W → ErrorPoly n)
    (hμ : ∀ c j k, polyNorm (substitution (sigma k).toNat (hodd k) (μ c j)) ≤ 1)
    {X : Tup (ExactMatrix q n (1 * 1) (C * m')) 6} {B : Nat}
    (hcol : ∀ c, EncRel sm (blkT X c) (fun j ↦ ∑ jj : Fin W, reducePoly q n (Xe (n := n) (e c jj)) •
      ∑ k : Fin P, branchTargetG (q := q) (p := p) sigma hodd (μ c jj) sp fam k j) B)
    {g' : ExactMatrix R n 1 m'} {Cw : Fin P → ExactPoly R n} {eB : Fin P → Fin 2 → ErrorMatrix n 1 m'}
    {BB : Nat}
    (hswitch : ∀ k j, reducePoly R n (substitution (sigma k).toNat (hodd k) sp) • fam 0 j k + fam 1 j k =
      (Cw k * reducePoly R n (1 * unitPair sp j)) • g' + reduceMatrix R n 1 m' (eB k j))
    (heB : ∀ k j, CoeffBound (eB k j) BB)
    (ν : Fin C → Fin W → ErrorPoly n)
    (hwalsh : ∀ c jj, ∑ k, reducePoly R n (substitution (sigma k).toNat (hodd k) (μ c jj)) * Cw k =
      reducePoly R n (ν c jj))
    {basis : List Nat} {x : ExactMatrix R n (1 * 1) (C * m')}
    (h434 : blockModSwitchRuns basis 1 (X.get 5) x)
    {Qq : Int} {D : ExactMatrix R n m' 1} (hD : g' * D = matrixPolynomial [Qq])
    {DI : ErrorMatrix n m' 1} {BD : Nat} (hDI : D = reduceMatrix R n m' 1 DI) (hDb : CoeffBound DI BD)
    {final : ExactMatrix R n (1 * 1) (C * 1)} (hblk : ∀ c, blk final c = matrixMul (blk x c) D) :
    ∃ (Kf : Fin 2 → ExactMatrix R n 1 (C * 1)) (ef : ErrorMatrix n 1 (C * 1)),
      CoeffBound ef (m' * n * ((B + (n + 2) * (p / 2)) / p + W * P * (n * 1 * BB)) * BD) ∧
      final = maskPart (reducePoly R n sm) Kf - ((Qq : Int) : ExactPoly R n) •
        reduceMatrix R n 1 (C * 1) (fun _ l ↦ ∑ jj, Xe (n := n) (e (colOf l) jj) * ν (colOf l) jj) +
        reduceMatrix R n 1 (C * 1) ef ∧
      Kf = fun k _ l ↦ ((blockModSwitchG basis 1 (Enc.A (blkT X (colOf l)) k 1) : ExactMatrix R n 1 m') * D) 0 0 := by
  have hn0 : 0 < n := by omega
  have hrel : ∀ c, Rel sm (fun k j ↦ blockModSwitchG basis 1 (Enc.A (blkT X c) k j) : Fin 2 → Fin 2 → ExactMatrix R n 1 m')
      (fun j ↦ reducePoly R n (1 * (∑ jj, Xe (n := n) (e c jj) * ν c jj) * unitPair sp j) • g')
      ![blockModSwitchG basis 1 (Enc.c (blkT X c) 0), blk x c]
      ((B + (n + 2) * (p / 2)) / p + W * P * (n * 1 * BB)) := by
    intro c
    have hb := blockModSwitchRuns_blk h434 c
    refine fresh_block_down hn hq hq1 hR hp hsm sigma hodd fam (e c) (μ c) (hμ c) (hcol c) hswitch heB
      (ν c) (hwalsh c) (fun k j ↦ blockModSwitchRuns_fnG hb _) (fun j ↦ ?_)
    fin_cases j
    · exact blockModSwitchRuns_fnG hb _
    · show blockModSwitchRuns basis 1 (Enc.c (blkT X c) 1) (blk x c)
      simpa [Enc.c, blkT_get] using hb
  obtain ⟨Kf, ef, hef, hfin, hKf⟩ := fresh_row_gen hn0 hrel (fun c ↦ rfl)
    (w := fun c ↦ ∑ jj, Xe (n := n) (e c jj) * ν c jj) (fun c ↦ by
      show reducePoly R n (1 * (∑ jj, Xe (n := n) (e c jj) * ν c jj) * unitPair sp 1) • g' = _
      rw [show unitPair sp 1 = 1 from rfl, one_mul, mul_one])
    hD hDI hDb hblk
  exact ⟨Kf, ef, hef, hfin, hKf⟩

/-- The reduced fresh error of `fresh_form_gen` is that of `freshErr` when every member value is
the sampler value of its PRF output (`C` columns of `n` members, member `c n + j` at `X^j`). -/
theorem fresh_err_eq {R C : Nat} (val : Int → Int) (z : Nat → Int) (ν : Fin C → Fin n → ErrorPoly n)
    (hval : ∀ (c : Fin C) (j : Fin n), reducePoly R n (ν c j) = reducePoly R n ((val (z (c * n + j)) : Int) : ErrorPoly n)) :
    reduceMatrix R n 1 (C * 1) (fun _ l ↦ ∑ jj : Fin n, Xe (n := n) (jj : Nat) * ν (colOf l) jj) =
      reduceMatrix R n 1 (C * 1) (freshErr n (C * 1) val z) := by
  funext r l
  simp only [reduceMatrix, freshErr, map_sum, map_mul, hval]
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  simp [colOf]


/-! ## The mask carrier -/

set_option maxHeartbeats 16000000 in
/-- `fresh_block_down` over any finite family `ι` of placed members. -/
theorem block_down_gen (hn : 1 < n) {ι : Type} [Fintype ι] [DecidableEq ι] {q q' p m' P : Nat}
    (hq : q = p * q') (hq1 : 1 < q) (hq' : 1 < q') (hp : 0 < p) {sm sp Δ' : ErrorPoly n}
    (hsm : polyNorm sm ≤ 1) (sigma : Fin P → Int) (hodd : ∀ i, Odd (sigma i).toNat)
    (fam : Fin 2 → Fin 2 → Fin P → ExactMatrix q' n 1 m') (e : ι → Nat) (μ : ι → ErrorPoly n)
    (hμ : ∀ j k, polyNorm (substitution (sigma k).toNat (hodd k) (μ j)) ≤ 1)
    {A : Fin 2 → Fin 2 → ExactMatrix q n 1 m'} {c : Fin 2 → ExactMatrix q n 1 m'} {B : Nat}
    (hcol : Rel sm A (fun j ↦ ∑ jj : ι, reducePoly q n (Xe (n := n) (e jj)) •
      ∑ k : Fin P, branchTargetG (q := q) (p := p) sigma hodd (μ jj) sp fam k j) c B)
    {g' : ExactMatrix q' n 1 m'} {C : Fin P → ExactPoly q' n} {eB : Fin P → Fin 2 → ErrorMatrix n 1 m'}
    {BB : Nat}
    (hswitch : ∀ k j, reducePoly q' n (substitution (sigma k).toNat (hodd k) sp) • fam 0 j k + fam 1 j k =
      (C k * reducePoly q' n (Δ' * unitPair sp j)) • g' + reduceMatrix q' n 1 m' (eB k j))
    (heB : ∀ k j, CoeffBound (eB k j) BB) (ν : ι → ErrorPoly n)
    (hwalsh : ∀ jj, ∑ k, reducePoly q' n (substitution (sigma k).toNat (hodd k) (μ jj)) * C k =
      reducePoly q' n (ν jj))
    {basis : List Nat} {K' : Fin 2 → Fin 2 → ExactMatrix q' n 1 m'} {R' : Fin 2 → ExactMatrix q' n 1 m'}
    (hK' : ∀ k j, blockModSwitchRuns basis 1 (A k j) (K' k j))
    (hR' : ∀ j, blockModSwitchRuns basis 1 (c j) (R' j)) :
    Rel sm K' (fun j ↦ reducePoly q' n (Δ' * (∑ jj, Xe (n := n) (e jj) * ν jj) * unitPair sp j) • g') R'
      ((B + (n + 2) * (p / 2)) / p + Fintype.card ι * P * (n * 1 * BB)) := by
  have hn0 : 0 < n := by omega
  let a : ι × Fin P → ErrorPoly n := fun x ↦ Xe (n := n) (e x.1) * substitution (sigma x.2).toNat (hodd x.2) (μ x.1)
  let b : ι × Fin P → ErrorPoly n := fun x ↦ substitution (sigma x.2).toNat (hodd x.2) sp
  have hcol' := hcol.target_eq (t' := fun j ↦ ∑ x : ι × Fin P, scaleUpMatrix q p
      (reducePoly q' n (a x) • (reducePoly q' n (b x) • fam 0 j x.2 + fam 1 j x.2))) (by
    funext j
    rw [Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun jj _ ↦ ?_
    rw [Finset.smul_sum]
    refine Finset.sum_congr rfl fun k _ ↦ ?_
    simp only [branchTargetG, a, b]
    rw [smul_scaleUpMatrix_gen hn0 hq hq', smul_smul, ← map_mul])
  have hfin := lut_finish_sum_gen (ι := ι × Fin P) hn0 hq hq1 hq' hp hsm a b
    (fun x ↦ (polyNorm_Xe_mul_le hn _ _).trans (hμ x.1 x.2))
    (bsw := fun x k j ↦ fam k j x.2) hcol' (C := fun x ↦ C x.2) (Δ' := Δ')
    (fun x j ↦ hswitch x.2 j) (fun x j ↦ heB x.2 j)
    (ν := ∑ jj, Xe (n := n) (e jj) * ν jj) (by
      rw [Fintype.sum_prod_type, map_sum]
      refine Finset.sum_congr rfl fun jj _ ↦ ?_
      simp only [a, map_mul, mul_assoc, ← Finset.mul_sum]
      rw [hwalsh jj])
    hK' hR'
  refine hfin.mono (le_of_eq ?_)
  simp [Fintype.card_prod, Fintype.card_fin, Nat.mul_assoc]

set_option maxHeartbeats 16000000 in
/-- **The mask carrier** (`hg` of the combination): the member total `X` of `M` placed
sampler-with-offset members (member `o`: LUT on `μ o` plus the LUT on the offset input `μoff`,
placed at `X^(e o)`), its second component moved down (`h335`), is
`maskPart s K - β • g' + reduce eg` with `β = Σ_o X^(e o) (ν o + νoff o)`. -/
theorem mask_carrier_gen (hn : 1 < n) {q R p m' M P : Nat} (hq : q = p * R) (hq1 : 1 < q)
    (hR : 1 < R) (hp : 0 < p) {sm sp : ErrorPoly n} (hsm : polyNorm sm ≤ 1)
    (sigma : Fin P → Int) (hodd : ∀ i, Odd (sigma i).toNat)
    (fam : Fin 2 → Fin 2 → Fin P → ExactMatrix R n 1 m') (e : Fin M → Nat)
    (μ : Fin M → ErrorPoly n) (μoff : ErrorPoly n)
    (hμ : ∀ o k, polyNorm (substitution (sigma k).toNat (hodd k) (μ o)) ≤ 1)
    (hμoff : ∀ k, polyNorm (substitution (sigma k).toNat (hodd k) μoff) ≤ 1)
    {X : Tup (ExactMatrix q n 1 m') 6} {B : Nat}
    (hX : EncRel sm X (fun j ↦ ∑ o : Fin M, reducePoly q n (Xe (n := n) (e o)) •
      ((∑ k : Fin P, branchTargetG (q := q) (p := p) sigma hodd (μ o) sp fam k j) +
        ∑ k : Fin P, branchTargetG (q := q) (p := p) sigma hodd μoff sp fam k j)) B)
    {g' : ExactMatrix R n 1 m'} {Cw : Fin P → ExactPoly R n} {eB : Fin P → Fin 2 → ErrorMatrix n 1 m'}
    {BB : Nat}
    (hswitch : ∀ k j, reducePoly R n (substitution (sigma k).toNat (hodd k) sp) • fam 0 j k + fam 1 j k =
      (Cw k * reducePoly R n (1 * unitPair sp j)) • g' + reduceMatrix R n 1 m' (eB k j))
    (heB : ∀ k j, CoeffBound (eB k j) BB) (ν : Fin M → ErrorPoly n) (νoff : ErrorPoly n)
    (hwalsh : ∀ o, ∑ k, reducePoly R n (substitution (sigma k).toNat (hodd k) (μ o)) * Cw k =
      reducePoly R n (ν o))
    (hwoff : ∑ k, reducePoly R n (substitution (sigma k).toNat (hodd k) μoff) * Cw k =
      reducePoly R n νoff)
    {basis : List Nat} {g : ExactMatrix R n 1 m'} (h335 : blockModSwitchRuns basis 1 (X.get 5) g) :
    ∃ (K : Fin 2 → ExactMatrix R n 1 m') (eg : ErrorMatrix n 1 m'),
      CoeffBound eg ((B + (n + 2) * (p / 2)) / p + M * 2 * P * (n * 1 * BB)) ∧
      g = maskPart (reducePoly R n sm) K -
        reducePoly R n (∑ o, Xe (n := n) (e o) * (ν o + νoff)) • g' + reduceMatrix R n 1 m' eg ∧
      K = fun k ↦ blockModSwitchG basis 1 (Enc.A X k 1) := by
  have hX' : Rel sm (Enc.A X) (fun j ↦ ∑ x : Fin M × Fin 2, reducePoly q n (Xe (n := n) (e x.1)) •
      ∑ k : Fin P, branchTargetG (q := q) (p := p) sigma hodd (![μ x.1, μoff] x.2) sp fam k j) (Enc.c X) B := by
    refine hX.target_eq ?_
    funext j
    rw [Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun o _ ↦ ?_
    simp only [Fin.sum_univ_two, Matrix.cons_val_zero, Matrix.cons_val_one, Matrix.head_cons, smul_add]
  let μ2 : Fin M × Fin 2 → ErrorPoly n := fun x ↦ ![μ x.1, μoff] x.2
  let ν2 : Fin M × Fin 2 → ErrorPoly n := fun x ↦ ![ν x.1, νoff] x.2
  have hμ2 : ∀ (x : Fin M × Fin 2) k, polyNorm (substitution (sigma k).toNat (hodd k) (μ2 x)) ≤ 1 := by
    rintro ⟨o, b⟩ k
    match b with
    | 0 => exact hμ o k
    | 1 => exact hμoff k
  have hw2 : ∀ x : Fin M × Fin 2, ∑ k, reducePoly R n (substitution (sigma k).toNat (hodd k) (μ2 x)) * Cw k =
      reducePoly R n (ν2 x) := by
    rintro ⟨o, b⟩
    match b with
    | 0 => exact hwalsh o
    | 1 => exact hwoff
  have hR2 : ∀ j, blockModSwitchRuns basis 1 (Enc.c X j)
      (![blockModSwitchG basis 1 (Enc.c X 0), g] j : ExactMatrix R n 1 m') := by
    intro j
    match j with
    | 0 => exact blockModSwitchRuns_fnG h335 _
    | 1 => exact h335
  have hrel := block_down_gen (ι := Fin M × Fin 2) (Δ' := 1) hn hq hq1 hR hp hsm sigma hodd fam
    (fun x ↦ e x.1) μ2 hμ2 hX' hswitch heB ν2 hw2
    (K' := fun k j ↦ blockModSwitchG basis 1 (Enc.A X k j))
    (fun k j ↦ blockModSwitchRuns_fnG h335 _) hR2
  obtain ⟨er, her, hc⟩ := hrel
  have h1 : g = reducePoly R n sm • blockModSwitchG basis 1 (Enc.A X 0 1) +
      blockModSwitchG basis 1 (Enc.A X 1 1) -
      reducePoly R n (1 * (∑ x : Fin M × Fin 2, Xe (n := n) (e x.1) * ![ν x.1, νoff] x.2) * unitPair sp 1) • g' +
      reduceMatrix R n 1 m' (er 1) := hc 1
  have hsum : (∑ x : Fin M × Fin 2, Xe (n := n) (e x.1) * ![ν x.1, νoff] x.2) =
      ∑ o, Xe (n := n) (e o) * (ν o + νoff) := by
    rw [Fintype.sum_prod_type]
    refine Finset.sum_congr rfl fun o _ ↦ ?_
    rw [Fin.sum_univ_two, mul_add]
    rfl
  refine ⟨fun k ↦ blockModSwitchG basis 1 (Enc.A X k 1), er 1, ?_, ?_, rfl⟩
  · refine coeffBound_mono (her 1) (le_of_eq ?_)
    rw [Fintype.card_prod, Fintype.card_fin, Fintype.card_fin]
  · rw [h1, hsum, show unitPair sp 1 = 1 from rfl, one_mul, mul_one]
    rfl

/-! ## Member values -/

/-- A LUT member's value as a sampled table entry (`sampledG`). -/
theorem sampled_of_walsh {q inverse s log : Nat} (hn : n = s * 2 ^ log) (hnpos : 0 < n)
    (T : Fin (2 ^ log) → ErrorPoly n)
    (hinverse : (inverse : ExactPoly q n) * ((2 ^ log : Nat) : ExactPoly q n) = 1)
    (index : Fin (2 ^ log) → Nat) (hodd : ∀ i, Odd (index i)) (perm : Fin (2 ^ log) ≃ Fin (2 ^ log))
    (hindex : ∀ i, index i % (2 * 2 ^ log) = 1 + 2 * 1 * (perm i).val)
    {C : Fin (2 ^ log) → ExactPoly q n}
    (hC : ∀ i, C i = lutFourierCoefficient (2 ^ log) (X q ^ s) (inverse : ExactPoly q n)
      (fun k ↦ reducePoly q n (T k)) (index i))
    {μ ν : ErrorPoly n} {z : Nat} (hμ : reducePoly q n μ = reducePoly q n (Xe (n := n) (s * z)))
    (hwalsh : ∑ i, reducePoly q n (substitution (index i) (hodd i) μ) * C i = reducePoly q n ν) :
    haveI : NeZero (2 ^ log) := ⟨by positivity⟩
    reducePoly q n ν = reducePoly q n (sampledG (2 ^ log) T z) := by
  rw [lut_value_of_walsh hn hnpos T hinverse index hodd perm hindex hC hμ hwalsh]
  rfl

/-- **The mask payload** in the form of `GenFinalInst.mask_inst`'s `hβ`: members placed at `X^o`,
member values the sampled bins of the readouts of `zN o` and the offset the sampled bin of `1`. -/
theorem mask_beta_eq {R P W' : Nat} [NeZero P] (T : Fin P → ErrorPoly n) (zN : Fin n → Nat)
    (ν : Fin n → ErrorPoly n) (νoff : ErrorPoly n)
    (hval : ∀ o, reducePoly R n (ν o) = reducePoly R n (sampledG P T (readoutExpG P W' (zN o))))
    (hoff : reducePoly R n νoff = reducePoly R n (sampledG P T 1)) :
    reducePoly R n (∑ o : Fin n, Xe (n := n) (o : Nat) * (ν o + νoff)) =
      reducePoly R n (∑ o : Fin n, Xe (n := n) o *
        (sampledG P T (readoutExpG P W' (zN o)) + sampledG P T 1)) := by
  simp only [map_sum, map_mul, map_add, hval, hoff]

/-- A one-hot selector row picks one cell: `Σ_k s(oW+k) a(t(oW+k)) = a(t(oW + sel))`. -/
theorem onehot_pick {W : Nat} (a : Nat → Int) (s t : Nat → Int) (o : Nat) (sel : Fin W)
    (hs : ∀ k : Fin W, s (o * W + k) = if k = sel then 1 else 0) :
    ∑ k ∈ Finset.range W, s (o * W + k) * a (t (o * W + k)).toNat = a (t (o * W + sel)).toNat := by
  rw [Finset.sum_range]
  simp only [hs, ite_mul, one_mul, zero_mul, Finset.sum_ite_eq', Finset.mem_univ, if_true]

/-- **The PRF output** of a member with one-hot selector rows in every bucket: `prfZB` is the sum
of the label exponents read at the selected cells. -/
theorem prfZB_onehot {W B : Nat} (a : Nat → Int) (s t : Fin B → Nat → Int) (o : Nat) (sel : Fin B → Fin W)
    (hs : ∀ b (k : Fin W), s b (o * W + k) = if k = sel b then 1 else 0) :
    prfZB W a s t o = ∑ b : Fin B, a (t b (o * W + sel b)).toNat := by
  unfold prfZB
  exact Finset.sum_congr rfl fun b _ ↦ onehot_pick a (s b) (t b) o (sel b) (hs b)


/-! ## Bridges from D's column / member totals -/

/-- `column_total_rel`'s target of placed members (`t f = X^(f mod n) • S f`, `W = n` members per
column) in the form of `fresh_form_gen`'s `hcol`. -/
theorem hcol_of_total {q m F C : Nat} {sm : ErrorPoly n} {x : Tup (ExactMatrix q n 1 m) 6} {B : Nat}
    (S : Fin F → Fin 2 → ExactMatrix q n 1 m) (cc : Fin C) (hF : n * C ≤ F)
    (h : EncRel sm x (∑ j : Fin n, (fun (f : Fin F) (jj : Fin 2) ↦
      reducePoly q n (Xe (n := n) (f.val % n)) • S f jj)
        ⟨n * cc + j, by have := cc.isLt; have := j.isLt; nlinarith⟩) B) :
    EncRel sm x (fun jj ↦ ∑ j : Fin n, reducePoly q n (Xe (n := n) (j : Nat)) •
      S ⟨n * cc + j, by have := cc.isLt; have := j.isLt; nlinarith⟩ jj) B := by
  unfold EncRel at h ⊢
  refine h.target_eq ?_
  funext jj
  rw [Finset.sum_apply]
  refine Finset.sum_congr rfl fun j _ ↦ ?_
  have hj := j.isLt
  have : (n * cc + j) % n = j := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt hj]
  simp only [this]

/-- `member_total_rel`'s target of `n` placed members (`t o = X^(o mod n) • (S o + S')`) in the
form of `mask_carrier_gen`'s `hX`. -/
theorem hX_of_total {q m : Nat} {sm : ErrorPoly n} {x : Tup (ExactMatrix q n 1 m) 6} {B : Nat}
    (S : Fin n → Fin 2 → ExactMatrix q n 1 m) (S' : Fin 2 → ExactMatrix q n 1 m)
    (h : EncRel sm x (∑ o : Fin n, fun jj ↦ reducePoly q n (Xe (n := n) (o.val % n)) • (S o jj + S' jj)) B) :
    EncRel sm x (fun jj ↦ ∑ o : Fin n, reducePoly q n (Xe (n := n) (o : Nat)) • (S o jj + S' jj)) B := by
  unfold EncRel at h ⊢
  refine h.target_eq ?_
  funext jj
  rw [Finset.sum_apply]
  refine Finset.sum_congr rfl fun o _ ↦ ?_
  simp only [Nat.mod_eq_of_lt o.isLt]


end RefreshTopProof.GenFinal
