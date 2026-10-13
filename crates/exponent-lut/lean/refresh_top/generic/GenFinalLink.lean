import GenFinal
import GenLut

/-!
# The action and the cross-modulus link of the final combination, for any parameter set

* `action_gen`, `action_chain_gen`: the target component's action. With `R = p M_x`, the target
  `(p x) g` (`g` the centered rebase of the gadget of `M_x`, any base and digit count) times the
  compact digits `Y` of `X = c' · lift(w)` (`w` over the CRT prime) is `Q_q T_x` with
  `T_x = reduce_R ((k x) · lift w)`, whenever `p c' = Q_q k`.
* `rho`: the residue modulo `q` of a polynomial over any multiple `R` of `q` (`reduce_q ∘
  canonicalLift`), a ring map on residues (`rho_add`, `rho_mul`, …, `rho_reduce`, `rho_rho`,
  `rho_centeredLift`); `residueEq_of_rho` turns entrywise `rho` equalities into `ResidueEq`.
* `tx_rho_gen`: the residue of `T_x` is `(k x) · w`; with `w = rho(G') · s₉ · s₁₀` (the export's
  chain) and one scalar identity modulo `q`, it is the residue of the decode stage's subtracted
  term `Δ r (c X^e) G'`.
* `link_gen`: the residues of `Y + E'` (`Y = β A'₀ + A'₁ - T_x` over `R`) agree with those of
  `decodeExpected A σ Δ r c e G + E'` (over `Q`) when `A'`, `β` and `T_x` agree with `A`, `σ` and
  the subtracted term modulo `q`: the `hyv` of `GenFinal.final_gen`.
-/

namespace RefreshTopProof.GenFinal

open Mxx.Primitives MxxRuntime MxxExponentLut Mxx.ExponentLut.Refresh RefreshTopProof GenDecode Gen
open scoped BigOperators

variable {n : Nat}

/-! ## The action of the target component -/

/-- The scaled one-row action at any ring dimension, base and digit count:
`(p x) g · D = scaleUp ((x) X)` for `g` the rebase of the gadget `g'` of `q'` and `D` the
compact rebase of the digits `D'` of `X` (`q = p q'`). -/
theorem action_scaleUp_gen (hn : 0 < n) {q p q' m m' : Nat} (hq : q = p * q') (hq' : 1 < q')
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q')
    (hlayout : backend.regularLayout q' n = some layout) (hexact : layout.droppedModuli = 0)
    {g' : ExactMatrix q' n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix q n 1 m} (hg : centeredRebaseRuns g' g)
    {X : ExactMatrix q' n 1 m'} {D' : ExactMatrix q' n m m'}
    (hD' : gadgetDecomposeRuns backend base digits X D')
    {D : ExactMatrix q n m m'} {bound : Nat} (hD : compactCenteredRebaseRuns bound D' D)
    (x : ErrorPoly n) :
    (reducePoly q n (((p : Int) : ErrorPoly n) * x) • g) * D =
      scaleUpMatrix q p (reducePoly q' n x • X) := by
  let G : ErrorMatrix n 1 m := fun row column ↦ centeredIntLift (g' row column)
  let E : ErrorMatrix n m m' := fun row column ↦ centeredIntLift (D' row column)
  have hgq : g = reduceMatrix q n 1 m G := centeredRebaseRuns_reduce hg
  have hDq : D = reduceMatrix q n m m' E := centeredRebaseRuns_reduce hD.2.2.1
  have hgq' : g' = reduceMatrix q' n 1 m G := (reduceMatrix_centeredIntLift hq' hn g').symm
  have hDq' : D' = reduceMatrix q' n m m' E := (reduceMatrix_centeredIntLift hq' hn D').symm
  have hrec := gadget_mul_decompose hq' hn layout hlayout hexact hg' hD'
  rw [← hrec, hgq, hDq, hgq', hDq']
  simp only [← reduceMatrix_smul, ← reduceMatrix_mul]
  funext row column
  simp only [scaleUpMatrix, reduceMatrix]
  rw [scaleUp_reducePoly hq hq' hn]
  congr 1
  simp only [Matrix.smul_mul, Matrix.smul_apply, smul_eq_mul, mul_assoc]

set_option maxHeartbeats 4000000 in
/-- **The action identity** `Pₓ Y = Q_q Tₓ` of the target component, for `X = reduce (c' W)`
and `p c' = Q_q k`: `(p x) g · Y = Q_q · reduce ((k x) W)`. -/
theorem action_gen (hn : 0 < n) {R Mx p m c : Nat} (hRp : R = p * Mx) (hMx : 1 < Mx)
    {backend : BackendContext} {base digits : Int} (LX : RegularLayout Mx)
    (hLX : backend.regularLayout Mx n = some LX) (hxX : LX.droppedModuli = 0)
    {Qq c' k : Int} (hc : (p : Int) * c' = Qq * k)
    {g' : ExactMatrix Mx n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix R n 1 m} (hg : centeredRebaseRuns g' g)
    {W : ErrorMatrix n 1 c} {X : ExactMatrix Mx n 1 c}
    (hX : X = reduceMatrix Mx n 1 c (((c' : Int) : ErrorPoly n) • W))
    {Y' : ExactMatrix Mx n m c} (hY' : gadgetDecomposeRuns backend base digits X Y')
    {Y : ExactMatrix R n m c} {bound : Nat} (hY : compactCenteredRebaseRuns bound Y' Y)
    (x : ErrorPoly n) :
    (reducePoly R n (((p : Int) : ErrorPoly n) * x) • g) * Y =
      ((Qq : Int) : ExactPoly R n) • reduceMatrix R n 1 c ((((k : Int) : ErrorPoly n) * x) • W) := by
  refine (action_scaleUp_gen hn hRp hMx LX hLX hxX hg' hg hY' hY x).trans ?_
  have h : ((p : Int) : ErrorPoly n) * (c' : ErrorPoly n) = (Qq : ErrorPoly n) * (k : ErrorPoly n) := by
    exact_mod_cast congrArg (fun z : Int ↦ (z : ErrorPoly n)) hc
  funext r col
  rw [hX]
  simp only [scaleUpMatrix, Matrix.smul_apply, smul_eq_mul, reduceMatrix_apply]
  rw [← map_mul, scaleUp_reducePoly hRp hMx hn, ← reducePoly_intCast (n := n) R Qq, ← map_mul]
  congr 1
  calc ((p : Int) : ErrorPoly n) * (x * ((c' : ErrorPoly n) * W r col))
      = (((p : Int) : ErrorPoly n) * (c' : ErrorPoly n)) * (x * W r col) := by ring
    _ = ((Qq : ErrorPoly n) * (k : ErrorPoly n)) * (x * W r col) := by rw [h]
    _ = (Qq : ErrorPoly n) * ((k : ErrorPoly n) * x * W r col) := by ring

/-- `X = (rebase w) · c'` (the export's `w_357 = w_355 · s₁₁`, `w_355` the rebase of `w_354`) is
`reduce (c' · lift w)`. -/
theorem action_input_gen {q Mx c : Nat} {c' : Int} {w : ExactMatrix q n 1 c}
    {w' : ExactMatrix Mx n 1 c} (hw : centeredRebaseRuns w w') {X : ExactMatrix Mx n 1 c}
    (hX : X = matrixMulScalarRight w' (matrixPolynomial [c'])) :
    X = reduceMatrix Mx n 1 c (((c' : Int) : ErrorPoly n) • fun r k ↦ centeredIntLift (w r k)) := by
  rw [hX, matrixMulScalarRight_eq, centeredRebaseRuns_reduce hw, reduceMatrix_smul]
  funext r k
  simp only [Matrix.smul_apply, smul_eq_mul, matrixPolynomial_single, reduceMatrix,
    reducePoly_intCast]

/-- **The target component's action, from the export's atoms**: `Pₓ = (p x) g`,
`X = rebase(w) · c'`, `Y` the compact digits of `X`; then `Pₓ Y = Q_q Tₓ` with
`Tₓ = reduce_R ((k x) · lift w)`. -/
theorem action_chain_gen (hn : 0 < n) {R Mx q p m c : Nat} (hRp : R = p * Mx) (hMx : 1 < Mx)
    {backend : BackendContext} {base digits : Int} (LX : RegularLayout Mx)
    (hLX : backend.regularLayout Mx n = some LX) (hxX : LX.droppedModuli = 0)
    {Qq c' k : Int} (hc : (p : Int) * c' = Qq * k)
    {g' : ExactMatrix Mx n 1 m} (hg' : gadgetMatrixRuns backend base digits g')
    {g : ExactMatrix R n 1 m} (hg : centeredRebaseRuns g' g)
    {w : ExactMatrix q n 1 c} {w' : ExactMatrix Mx n 1 c} (hw : centeredRebaseRuns w w')
    {X : ExactMatrix Mx n 1 c} (hX : X = matrixMulScalarRight w' (matrixPolynomial [c']))
    {Y' : ExactMatrix Mx n m c} (hY' : gadgetDecomposeRuns backend base digits X Y')
    {Y : ExactMatrix R n m c} {bound : Nat} (hY : compactCenteredRebaseRuns bound Y' Y)
    (x : ErrorPoly n) :
    (reducePoly R n (((p : Int) : ErrorPoly n) * x) • g) * Y =
      ((Qq : Int) : ExactPoly R n) • reduceMatrix R n 1 c ((((k : Int) : ErrorPoly n) * x) •
        fun r col ↦ centeredIntLift (w r col)) :=
  action_gen hn hRp hMx LX hLX hxX hc hg' hg (action_input_gen hw hX) hY' hY x

/-! ## Residues modulo a CRT prime -/

/-- The residue modulo `q` of a polynomial over any modulus. -/
noncomputable def rho (q : Nat) {R : Nat} (x : ExactPoly R n) : ExactPoly q n :=
  reducePoly q n (canonicalLift x)

section Rho

variable {q R : Nat}

theorem rho_coeff (hn : 0 < n) (hq : 1 < q) (x : ExactPoly R n) (i : Fin n) :
    (rho q x).coeff i = ((x.coeff i).val : ZMod q) := by
  rw [rho, reducePoly_coeff hq hn, canonicalLift_coeff hn]
  push_cast
  rfl

theorem rho_reduce (hn : 0 < n) (hq : 1 < q) (hR : 1 < R) (hqR : q ∣ R) (z : ErrorPoly n) :
    rho q (reducePoly R n z) = reducePoly q n z := by
  haveI : NeZero R := ⟨by omega⟩
  haveI : Fact (1 < q) := ⟨hq⟩
  apply negacyclic_ext hn
  intro i
  rw [rho_coeff hn hq, reducePoly_coeff hR hn, reducePoly_coeff hq hn, val_cast_eq_cast,
    ZMod.cast_intCast hqR]

theorem rho_eq_of (hn : 0 < n) (hq : 1 < q) (hR : 1 < R) (hqR : q ∣ R) (x : ExactPoly R n) :
    rho q x = reducePoly q n (canonicalLift x) := rfl

theorem rho_add (hn : 0 < n) (hq : 1 < q) (hR : 1 < R) (hqR : q ∣ R) (x y : ExactPoly R n) :
    rho q (x + y) = rho q x + rho q y := by
  conv_lhs => rw [← reducePoly_canonicalLift hR hn x, ← reducePoly_canonicalLift hR hn y,
    ← map_add, rho_reduce hn hq hR hqR, map_add]
  rfl

theorem rho_sub (hn : 0 < n) (hq : 1 < q) (hR : 1 < R) (hqR : q ∣ R) (x y : ExactPoly R n) :
    rho q (x - y) = rho q x - rho q y := by
  conv_lhs => rw [← reducePoly_canonicalLift hR hn x, ← reducePoly_canonicalLift hR hn y,
    ← map_sub, rho_reduce hn hq hR hqR, map_sub]
  rfl

theorem rho_mul (hn : 0 < n) (hq : 1 < q) (hR : 1 < R) (hqR : q ∣ R) (x y : ExactPoly R n) :
    rho q (x * y) = rho q x * rho q y := by
  conv_lhs => rw [← reducePoly_canonicalLift hR hn x, ← reducePoly_canonicalLift hR hn y,
    ← map_mul, rho_reduce hn hq hR hqR, map_mul]
  rfl

theorem rho_intCast (hn : 0 < n) (hq : 1 < q) (hR : 1 < R) (hqR : q ∣ R) (c : Int) :
    rho q ((c : ExactPoly R n)) = (c : ExactPoly q n) := by
  rw [← reducePoly_intCast, rho_reduce hn hq hR hqR, reducePoly_intCast]

/-- Residues pass through an intermediate multiple: `q ∣ R`, `x` over `Q` with `R ∣ Q`. -/
theorem rho_rho (hn : 0 < n) (hq : 1 < q) (hR : 1 < R) (hqR : q ∣ R) {Q : Nat} (x : ExactPoly Q n) :
    rho q (rho R x) = rho q x := by
  show rho q (reducePoly R n (canonicalLift x)) = _
  rw [rho_reduce hn hq hR hqR]
  rfl

/-- The residue of a centered lift re-encoded over a multiple of `q`. -/
theorem rho_centeredLift (hn : 0 < n) (hq : 1 < q) {P : Nat} (hP : 1 < P) (hqP : q ∣ P)
    (x : ExactPoly P n) : reducePoly q n (centeredIntLift x) = rho q x := by
  conv_rhs => rw [← reducePoly_centeredIntLift hP hn x]
  rw [rho_reduce hn hq hP hqP]

/-- The residue modulo `q` of a polynomial over `q` itself. -/
theorem rho_self (hn : 0 < n) (hq : 1 < q) (x : ExactPoly q n) : rho q x = x :=
  reducePoly_canonicalLift hq hn x

/-- An ordinary modulus reduction is the residue map. -/
theorem modulusReduce_rho (hn : 0 < n) {P rows cols : Nat} {x : ExactMatrix R n rows cols}
    {y : ExactMatrix P n rows cols} (h : modulusReduceRuns x y) (r : Fin rows) (c : Fin cols) :
    y r c = rho P (x r c) := by
  have hP : 1 < P := h.1
  haveI : Fact (1 < P) := ⟨hP⟩
  rw [h.2.2]
  apply negacyclic_ext hn
  intro i
  rw [rho_coeff hn hP, modulusReduce, polynomialOfCoefficients_coeff hP hn]
  simp

/-- A centered rebase between multiples of `q` keeps the residues. -/
theorem centeredRebase_rho (hn : 0 < n) (hq : 1 < q) {P Pc rows cols : Nat} (hP : 1 < P)
    (hqP : q ∣ P) (hqPc : q ∣ Pc) {x : ExactMatrix P n rows cols} {y : ExactMatrix Pc n rows cols}
    (h : centeredRebaseRuns x y) (r : Fin rows) (c : Fin cols) : rho q (y r c) = rho q (x r c) := by
  have hPc : 1 < Pc := h.1
  rw [centeredRebaseRuns_reduce h]
  simp only
  rw [rho_reduce hn hq hPc hqPc, rho_centeredLift hn hq hP hqP]

end Rho

/-- Entrywise equal residues are equal `ResidueEq` residues. -/
theorem residueEq_of_rho (hn : 0 < n) {q R Q m : Nat} (hq : 1 < q) {y : ExactMatrix R n 1 m}
    {v : ExactMatrix Q n 1 m} (h : ∀ c, rho q (y 0 c) = rho q (v 0 c)) : ResidueEq q y v := by
  intro c i
  rw [← rho_coeff hn hq, ← rho_coeff hn hq, h c]

/-! ## The residue of the action's target -/

/-- **The residue of `Tₓ`.** With `Tₓ = reduce_R ((k x) · lift w)` (`w` over `q ∣ R`),
`w = w₀ · s₉ · s₁₀` and `w₀` the reduction of `G'` (the rebased gadget over `Q`), the residue of
`Tₓ` is `(k x) s₉ s₁₀ ρ(G')`; one scalar identity modulo `q` makes it the residue of `T G'`. -/
theorem tx_rho_gen (hn : 0 < n) {q R Q c : Nat} (hq : 1 < q) (hR : 1 < R) (hqR : q ∣ R)
    (hQ : 1 < Q) (hqQ : q ∣ Q) {k : Int} {x : ErrorPoly n} {s9 s10 : Int}
    {G' : ExactMatrix Q n 1 c} {w₀ w₁ w : ExactMatrix q n 1 c} (hw₀ : modulusReduceRuns G' w₀)
    (hw₁ : w₁ = matrixMulScalarRight w₀ (matrixPolynomial [s9]))
    (hw : w = matrixMulScalarRight w₁ (matrixPolynomial [s10]))
    {T : ExactPoly Q n}
    (hscalar : reducePoly q n (((k : Int) : ErrorPoly n) * x) * (s9 : ExactPoly q n) *
      (s10 : ExactPoly q n) = rho q T)
    (col : Fin c) :
    rho q ((reduceMatrix R n 1 c ((((k : Int) : ErrorPoly n) * x) •
        fun r col ↦ centeredIntLift (w r col))) 0 col) = rho q (T * G' 0 col) := by
  rw [reduceMatrix, Matrix.smul_apply, smul_eq_mul, rho_reduce hn hq hR hqR, map_mul,
    rho_centeredLift hn hq hq (dvd_refl q), rho_self hn hq, hw, hw₁, matrixMulScalarRight_eq,
    matrixMulScalarRight_eq, Matrix.smul_apply, Matrix.smul_apply, smul_eq_mul, smul_eq_mul,
    matrixPolynomial_single, matrixPolynomial_single, modulusReduce_rho hn hw₀,
    rho_mul hn hq hQ hqQ, ← hscalar]
  ring

/-- The scalar identity of `tx_rho_gen` from the instance constants: `k s₁₀ ≡ 1` and `s₉ ≡ Δ`
modulo `q`, and the target component's payload `x` has the residue of `r (c X^e)`. -/
theorem scalar_gen (hn : 0 < n) {q Q : Nat} (hq : 1 < q) (hQ : 1 < Q) (hqQ : q ∣ Q)
    {k s9 s10 Δ : Int} (hk : k * s10 ≡ 1 [ZMOD q]) (h9 : s9 ≡ Δ [ZMOD q])
    {x : ErrorPoly n} {u : ExactPoly Q n} (hx : reducePoly q n x = rho q u) :
    reducePoly q n (((k : Int) : ErrorPoly n) * x) * (s9 : ExactPoly q n) * (s10 : ExactPoly q n) =
      rho q ((Δ : ExactPoly Q n) * u) := by
  have cong : ∀ a b : Int, a ≡ b [ZMOD q] → (a : ExactPoly q n) = (b : ExactPoly q n) := by
    intro a b h
    rw [← map_intCast (algebraMap (ZMod q) (ExactPoly q n)) a,
      ← map_intCast (algebraMap (ZMod q) (ExactPoly q n)) b, (ZMod.intCast_eq_intCast_iff _ _ _).2 h]
  rw [rho_mul hn hq hQ hqQ, rho_intCast hn hq hQ hqQ, map_mul, hx, reducePoly_intCast, cong _ _ h9]
  calc (k : ExactPoly q n) * rho q u * (Δ : ExactPoly q n) * (s10 : ExactPoly q n)
      = (Δ : ExactPoly q n) * rho q u * ((k * s10 : Int) : ExactPoly q n) := by push_cast; ring
    _ = (Δ : ExactPoly q n) * rho q u := by rw [cong _ _ hk]; simp

/-! ## The cross-modulus link -/

/-- The decode stage's subtracted term of column `col` (`(Δ r) (c X^e) G'`). -/
noncomputable def decodeTerm {Q Qg m : Nat} (Δ : Int) (r : ExactMatrix Q n 1 1) (c e : Int)
    (G : ExactMatrix Qg n 1 m) (col : Fin m) : ExactPoly Q n :=
  (Δ : ExactPoly Q n) * r 0 0 * multiplyMonomial (liftInteger c) e 0 0 * centeredRebase G 0 col

/-- **The cross-modulus link** (`hyv` of `final_gen`). Over `R` the value is
`Y = β A'₀ + A'₁ - Tₓ`; the decode stage expects `σ A₀ + A₁ - (Δ r)(c X^e) G'` over `Q`. When
`A'`, `β`, `Tₓ` have the residues of `A`, `σ` and the subtracted term modulo `q` (`q ∣ R`,
`q ∣ Q`), `Y + E'` and `decodeExpected … + E'` have equal residues. -/
theorem link_gen (hn : 0 < n) {q R Q Qg m : Nat} (hq : 1 < q) (hR : 1 < R) (hqR : q ∣ R)
    (hQ : 1 < Q) (hqQ : q ∣ Q)
    {β : ExactPoly R n} {A' : ExactMatrix R n (1 + 1) m} {Tx : ExactMatrix R n 1 m}
    {A : ExactMatrix Q n 2 m} {σ : ExactMatrix Q n 1 1} {Δ c e : Int} {r : ExactMatrix Q n 1 1}
    {G : ExactMatrix Qg n 1 m}
    (hA : ∀ (i : Fin 2) col, rho q (A' i col) = rho q (A i col))
    (hβ : rho q β = rho q (σ 0 0))
    (hT : ∀ col, rho q (Tx 0 col) = rho q (decodeTerm Δ r c e G col))
    (E : ErrorMatrix n 1 m) :
    ResidueEq q (β • rowOf A' 0 + rowOf A' 1 - Tx + reduceMatrix R n 1 m E)
      (decodeExpected A σ Δ r c e G + reduceMatrix Q n 1 m E) := by
  refine residueEq_add_reduce hn hR hQ hqR hqQ (residueEq_of_rho hn hq fun col ↦ ?_) E
  simp only [Matrix.sub_apply, Matrix.add_apply, Matrix.smul_apply, smul_eq_mul, rowOf]
  rw [rho_sub hn hq hR hqR, rho_add hn hq hR hqR, rho_mul hn hq hR hqR, hβ, hA 0 col, hA 1 col,
    hT col]
  simp only [decodeExpected, decodeTerm, matrixSub, matrixAdd, matrixMulScalarLeft,
    Matrix.sub_apply, Matrix.add_apply, matrixPolynomial_single]
  simp only [rho_sub hn hq hQ hqQ, rho_add hn hq hQ hqQ, rho_mul hn hq hQ hqQ]

/-! ## Equal decompositions in two stages -/

/-- Compact digits of equal targets are equal (any base, digit count, bounds). -/
theorem digits_unique {q p rows cols inner : Nat} {backend : BackendContext} {base digits : Int}
    {x x' : ExactMatrix q n rows cols} (hx : x = x') {Dz Dz' : ExactMatrix q n inner cols}
    (hD : gadgetDecomposeRuns backend base digits x Dz) (hD' : gadgetDecomposeRuns backend base digits x' Dz')
    {b b' : Nat} {D D' : ExactMatrix p n inner cols} (hC : compactCenteredRebaseRuns b Dz D)
    (hC' : compactCenteredRebaseRuns b' Dz' D') : D = D' := by
  subst hx
  obtain rfl := gadgetDecomposeRuns_unique hD hD'
  exact hC.2.2.1.2.trans hC'.2.2.1.2.symm

/-- **The combination digits of two stages agree**: the compact digits of `Q_q · reduce(A)` for the
same hash-sampled `A` and equal `Q_q`. -/
theorem comb_digits_link {Q R c inner : Nat} {backend : BackendContext} {base digits : Int}
    {A A' : ExactMatrix Q n 2 c} (hA : A = A') {Qq Qq' : Int} (hQ : Qq = Qq')
    {A1 A1' : ExactMatrix R n 2 c} (h1 : modulusReduceRuns A A1) (h1' : modulusReduceRuns A' A1')
    {Dz Dz' : ExactMatrix R n inner c}
    (hD : gadgetDecomposeRuns backend base digits (matrixMulScalarLeft (matrixPolynomial [Qq]) A1) Dz)
    (hD' : gadgetDecomposeRuns backend base digits (matrixMulScalarLeft (matrixPolynomial [Qq']) A1') Dz')
    {b b' : Nat} {D D' : ExactMatrix R n inner c} (hC : compactCenteredRebaseRuns b Dz D)
    (hC' : compactCenteredRebaseRuns b' Dz' D') : D = D' := by
  subst hA hQ
  obtain rfl := modulusReduceRuns_unique h1 h1'
  exact digits_unique rfl hD hD' hC hC'

/-- **The action digits of two stages agree**: the chain gadget → rebase → reduce → `· s₁ · s₂` →
rebase → `· c'` → digits → compact rebase, with equal constants. -/
theorem action_digits_link {Qg Q q Mx R c inner : Nat} {backend : BackendContext}
    {gb gd base digits : Int}
    {G G' : ExactMatrix Qg n 1 c} (hG : gadgetMatrixRuns backend gb gd G) (hG' : gadgetMatrixRuns backend gb gd G')
    {Gp Gp' : ExactMatrix Q n 1 c} (hp : centeredRebaseRuns G Gp) (hp' : centeredRebaseRuns G' Gp')
    {w w' : ExactMatrix q n 1 c} (hw : modulusReduceRuns Gp w) (hw' : modulusReduceRuns Gp' w')
    {s1 s1' s2 s2' c1 c1' : Int} (e1 : s1 = s1') (e2 : s2 = s2') (e3 : c1 = c1')
    {u u' : ExactMatrix Mx n 1 c}
    (hu : centeredRebaseRuns (matrixMulScalarRight (matrixMulScalarRight w (matrixPolynomial [s1]))
      (matrixPolynomial [s2])) u)
    (hu' : centeredRebaseRuns (matrixMulScalarRight (matrixMulScalarRight w' (matrixPolynomial [s1']))
      (matrixPolynomial [s2'])) u')
    {Yz Yz' : ExactMatrix Mx n inner c}
    (hY : gadgetDecomposeRuns backend base digits (matrixMulScalarRight u (matrixPolynomial [c1])) Yz)
    (hY' : gadgetDecomposeRuns backend base digits (matrixMulScalarRight u' (matrixPolynomial [c1'])) Yz')
    {b b' : Nat} {Y Y' : ExactMatrix R n inner c} (hC : compactCenteredRebaseRuns b Yz Y)
    (hC' : compactCenteredRebaseRuns b' Yz' Y') : Y = Y' := by
  subst e1 e2 e3
  obtain rfl := gadgetMatrixRuns_unique hG hG'
  obtain rfl := centeredRebaseRuns_unique hp hp'
  obtain rfl := modulusReduceRuns_unique hw hw'
  obtain rfl := centeredRebaseRuns_unique hu hu'
  exact digits_unique rfl hY hY' hC hC'

/-! ## The mask residue and the payload residue -/

/-- `polynomialOfCoefficients σ` is the reduction of the integer polynomial `σ`. -/
theorem polynomialOfCoefficients_reduce (hn : 0 < n) {Q : Nat} (hQ : 1 < Q) (σ : Fin n → Int) :
    (polynomialOfCoefficients σ : ExactPoly Q n) = reducePoly Q n (intPoly σ) := by
  haveI : Fact (1 < Q) := ⟨hQ⟩
  apply negacyclic_ext hn
  intro i
  rw [polynomialOfCoefficients_coeff hQ hn, reducePoly_coeff hQ hn, intPoly_coeff hn]

theorem sum_Xe_intPoly (σ : Fin n → Int) :
    ∑ o : Fin n, Xe (n := n) o * ((σ o : Int) : ErrorPoly n) = intPoly σ := by
  unfold intPoly
  exact Finset.sum_congr rfl fun o _ ↦ mul_comm _ _

/-- **The mask residue.** The mask payload `β = Σ_o X^o (V(z_o) + V(1))` of a binary sampler
(table `h` on every bin, `2 h ≡ 1 (mod q)`) has, modulo `q`, the residue of `σ` whenever
`σ_o = 1 - ⌊(z_o mod 2W') / W'⌋` (the requirement's mask formula). -/
theorem mask_rho_gen (hn : 0 < n) {q R Q P W' : Nat} [NeZero P] (hq : 1 < q) (hR : 1 < R)
    (hqR : q ∣ R) (hQ : 1 < Q) (hqQ : q ∣ Q) (hP : 0 < P / 2) (hP2 : P % 2 = 0) (hW : 0 < W')
    {h : Int} (h2 : ((2 * h : Int) : ZMod q) = 1) {zN : Fin n → Nat} {σ : Fin n → Int}
    (hσ : ∀ o, σ o = 1 - ((zN o : Int) % (2 * (W' : Int))) / (W' : Int)) {β : ExactPoly R n}
    (hβ : β = reducePoly R n (∑ o : Fin n, Xe (n := n) o *
      (sampledG P (binTable P fun _ ↦ h) (readoutExpG P W' (zN o)) + sampledG P (binTable P fun _ ↦ h) 1))) :
    rho q β = rho q (polynomialOfCoefficients σ : ExactPoly Q n) := by
  rw [hβ, rho_reduce hn hq hR hqR, polynomialOfCoefficients_reduce hn hQ, rho_reduce hn hq hQ hqQ,
    ← sum_Xe_intPoly, map_sum, map_sum]
  refine Finset.sum_congr rfl fun o _ ↦ ?_
  rw [map_mul, map_mul, member_sigma_gen hP hP2 hW h2, hσ o]

/-- **The payload residue** (`hx` of `final_inst`): when the target component's payload `x` has,
modulo `q`, the residue of `c X^e · lift(r)` (`r` the decode stage's factor), it has the residue
`final_inst` needs. The lift is centered, so no hypothesis on `r` is needed. -/
theorem payload_rho_gen (hn : 0 < n) {q Q : Nat} (hq : 1 < q) (hQ : 1 < Q) (hqQ : q ∣ Q)
    {r : ExactPoly Q n} {c e : Int} {x : ErrorPoly n}
    (hμ : reducePoly q n x = reducePoly q n (message c e * centeredIntLift r)) :
    reducePoly q n x = rho q (r * multiplyMonomial (liftInteger c : ExactMatrix Q n 1 1) e 0 0) := by
  rw [hμ, multiplyMonomial_liftInteger, rho_mul hn hq hQ hqQ, rho_reduce hn hq hQ hqQ,
    ← rho_centeredLift hn hq hQ hqQ, map_mul, mul_comm]

end RefreshTopProof.GenFinal
