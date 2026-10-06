import ExponentPairing

/-!
# Pairings with the numerators of `lwr_prf`

A weight on the error after a ModDown pairs with the numerator divided by the dropped factor. The
numerators of the cells, of the alignments, and of the LUT are sums of actions, scalings and
residues; their pairings split into pairings of pulled-back weights with every source: the
errors entering the operation, the residues of its rows, and the mask-weighted and plain residues
of its keys.
-/

namespace MxxExponentLut

open Mxx.Primitives MxxRuntime
open scoped BigOperators

variable {n : Nat}

/-- The rows `j` of a stacked matrix: the top half for `0`, the bottom half for `1`. -/
def halfRows {α : Type} {m m' : Nat} (d : Matrix (Fin (m + m)) (Fin m') α) (j : Fin 2) :
    Matrix (Fin m) (Fin m') α :=
  if j = 0 then topRows d else bottomRows d

theorem pairing_divInt (hn : 0 < n) {rows columns : Nat} (W : Weight n rows columns) {p : ℤ}
    (hp : p ≠ 0) {x : ErrorMatrix n rows columns}
    (hdiv : ∀ r c, x r c = (p : ErrorPoly n) * divInt p x r c) :
    pairing W (divInt p x) = pairing W x / p :=
  pairing_div W hp hdiv

/-- The pairing with a stacked action, as a sum over the two rows. -/
theorem pairing_pairMul_sum (hn : 0 < n) {m m' : Nat} (V : Weight n 1 m')
    (y : Fin 2 → ErrorMatrix n 1 m) (d : ErrorMatrix n (m + m) m') :
    pairing V (pairMul y d) = ∑ j', pairing (pullMul V (halfRows d j')) (y j') := by
  rw [pairing_pairMul hn, Fin.sum_univ_two]
  rfl

/-- The pairing with a cell numerator: the aligned member `va` after the public digits `Ds` and
the private digits `Dr`, the state error `es` times `α` after `Dr`, the row residue `X`, the
mask-weighted key residue `M₀`, and the key residue `M₁`. -/
theorem pairing_cellNumer (hn : 0 < n) {mA m m' : Nat} (V : Weight n 1 m')
    (va : Fin 2 → ErrorMatrix n 1 mA) (Ds : Fin 2 → ErrorMatrix n (mA + mA) m)
    (α : ErrorPoly n) (es : Fin 2 → ErrorMatrix n 1 m) (Dr : ErrorMatrix n (m + m) m')
    (sm : ErrorPoly n) (X M0 M1 : ErrorMatrix n 1 m') :
    pairing V (pairMul (fun j' ↦ pairMul va (Ds j') + α • es j') Dr - X + sm • M0 + M1) =
      (∑ j', ∑ k, pairing (pullMul (pullMul V (halfRows Dr j')) (halfRows (Ds j') k)) (va k)) +
        (∑ j', pairing (pullSmul (pullMul V (halfRows Dr j')) α) (es j')) -
        pairing V X + pairing (pullSmul V sm) M0 + pairing V M1 := by
  rw [pairing_add, pairing_add, pairing_sub, pairing_pairMul_sum hn, pairing_smul hn]
  simp only [pairing_add, pairing_smul hn, pairing_pairMul_sum hn, Finset.sum_add_distrib]

/-- The pairing with an alignment numerator: the member error after the digits `H`, the row
residue, the mask-weighted key residue, and the key residue. -/
theorem pairing_alignNumer (hn : 0 < n) {m m' : Nat} (U : Weight n 1 m')
    (ea : ErrorMatrix n 1 m) (H : ErrorMatrix n m m') (sm : ErrorPoly n) (R K0 K1 : ErrorMatrix n 1 m') :
    pairing U (ea * H - R + sm • K0 + K1) =
      pairing (pullMul U H) ea - pairing U R + pairing (pullSmul U sm) K0 + pairing U K1 := by
  rw [pairing_add, pairing_add, pairing_sub, pairing_mul hn, pairing_smul hn]

/-- The pairing with the error leaving a block: every cell's numerator divided by the dropped
factor, minus its RHS error times `c i`. -/
theorem pairing_blockErr (hn : 0 < n) {C m : Nat} (V : Weight n 1 m) {p : ℤ} (hp : p ≠ 0)
    (numer : Fin C → ErrorMatrix n 1 m)
    (hdiv : ∀ i r c, numer i r c = (p : ErrorPoly n) * divInt p (numer i) r c)
    (c : Fin C → ErrorPoly n) (eB : Fin C → ErrorMatrix n 1 m) :
    pairing V (∑ i, (divInt p (numer i) - c i • eB i)) =
      ∑ i, (pairing V (numer i) / p - pairing (pullSmul V (c i)) (eB i)) := by
  rw [pairing_sum]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  rw [pairing_sub, pairing_divInt hn V hp (hdiv i), pairing_smul hn]

end MxxExponentLut
