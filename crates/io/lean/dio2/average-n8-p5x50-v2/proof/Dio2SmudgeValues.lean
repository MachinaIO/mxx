import Dio2Generic
import GenInstLemmas
import Dio2Rel

/-!
# Multi-digit smudge values

The digit, column and coefficient indices remain quantified. The readout-stage obligation is
that each digit polynomial reduces to the packed odd integers. The weighted sum is taken in
the ring; gadget retargeting noise is accounted for separately by the encoding relation.
-/
namespace Dio2Proof
open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof.Gen RefreshTopProof.GenIface
open Mxx.ExponentLut.Refresh
open scoped BigOperators

/-- A three-component digit accumulator is a finite sum, without digit enumeration. -/
theorem online1_digit_fold {α : Type} [AddCommMonoid α] {count : Nat}
    {body : Nat → Tup α 3 → Tup α 3 → Prop} {final : Tup α 3}
    (term : Nat → Fin 3 → α)
    (hstep : ∀ i current next, i < count → body i current next →
      ∀ component, next.get component = current.get component + term i component)
    (hrun : MxxIR.IterRuns body count (0, (0, (0, ()))) final) :
    ∀ component, final.get component = ∑ i ∈ Finset.range count, term i component := by
  have result := MxxIR.IterRuns.invariant
    (Invariant := fun k state ↦ k ≤ count → ∀ component,
      state.get component = ∑ i ∈ Finset.range k, term i component)
    (fun _ component ↦ by fin_cases component <;> change 0 = _ <;> simp)
    (fun i current next previous step hk component ↦ by
      rw [hstep i current next (by omega) step component,
        previous (by omega) component, Finset.sum_range_succ]) hrun
  exact result le_rfl

/-- Packed readouts acted on by indexed weights retain their individual gadget errors.
The targets include every weight, and the resulting keys are the same indexed sums. -/
theorem online1_digit_encoding {n q m count : Nat} (hn : 0 < n) (hq : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {sm : ErrorPoly n} {g : ExactMatrix q n 1 m}
    (hg : gadgetMatrixRuns backend base digits g)
    (packed : Tup (ExactMatrix q n 1 (count * m)) 6)
    (target : Fin count → Fin 2 → ExactPoly q n) {B : Nat}
    (hpacked : ∀ r, EncRel sm (RefreshTopProof.GenInst.blkT packed r)
      (fun port ↦ target r port • g) B)
    (weight : Fin count → ExactMatrix q n 1 1)
    (decomposition : Fin count → ExactMatrix q n m 1)
    (hdecomposition : ∀ r, gadgetDecomposeRuns backend base digits (weight r) (decomposition r)) :
    ∃ error : ErrorMatrix n 1 1,
      CoeffBound error (count * (m * n * B * (base.toNat / 2))) ∧
      (∑ r, RefreshTopProof.GenInst.blk packed.2.2.2.2.2.1 r * decomposition r) =
        maskPart (reducePoly q n sm)
          ![∑ r, RefreshTopProof.GenInst.blk packed.2.1 r * decomposition r,
            ∑ r, RefreshTopProof.GenInst.blk packed.2.2.2.1 r * decomposition r] -
        (∑ r, target r 1 • weight r) + reduceMatrix q n 1 1 error := by
  have relations := fun r ↦ Rel.gamma hn hq layout hlayout hexact hg
    (hdecomposition r) (hpacked r)
  obtain ⟨error, herror, equation⟩ := Rel.sum Finset.univ (fun r _ ↦ relations r)
  refine ⟨error 1, by simpa using herror 1, ?_⟩
  simpa only [Finset.sum_apply, Finset.sum_const_zero, Enc.c, Enc.A,
    RefreshTopProof.GenInst.blkT, maskPart, Matrix.cons_val_zero,
    Matrix.cons_val_one] using equation 1

/-- A one-row gadget supplies canonical digits for every scalar weight. -/
theorem online1_weight_decomposition {n q m : Nat} (hn : 0 < n) (hq : 1 < q)
    {backend : BackendContext} {base digits : Int} (layout : RegularLayout q)
    (hlayout : backend.regularLayout q n = some layout) (hexact : layout.droppedModuli = 0)
    {g : ExactMatrix q n 1 m} (hg : gadgetMatrixRuns backend base digits g)
    (weight : ExactMatrix q n 1 1) :
    gadgetDecomposeRuns backend base digits weight (decG (inner := m) backend base digits weight) := by
  obtain ⟨actual, hactual, hbase, hdigits, hwidth, _⟩ := hg
  rw [hlayout] at hactual
  cases hactual
  obtain ⟨decomposition, hdecomposition⟩ :=
    gadgetDecomposeRuns_existsG hn layout hlayout hexact hq hbase hdigits hwidth weight
  rw [← decG_eq hdecomposition]
  exact hdecomposition

/-- The integer residual of a column after one coefficientwise half offset. -/
noncomputable def smudgeResidual {n d columns : Nat} (P : Nat)
    (digit : Fin (d + 1) → Fin columns → Fin n → Int) : ErrorMatrix n 1 columns :=
  fun _ c ↦ intPoly (fun i ↦ smudgeInt P (d + 1) (fun r ↦ digit r c i))

/-- The full residual bound is `P^(d+1)/2`, independent of the number of columns and n. -/
theorem smudgeResidual_bound {n P d columns : Nat} (hn : 0 < n) (hP : P % 2 = 0)
    (digit : Fin (d + 1) → Fin columns → Fin n → Int)
    (hodd : ∀ r c i, digit r c i % 2 = 1)
    (hbound : ∀ r c i, |digit r c i| ≤ (P : Int) - 1) :
    CoeffBound (smudgeResidual P digit) (P ^ (d + 1) / 2) := by
  intro row c i
  simp only [smudgeResidual, intPoly_coeff hn]
  exact smudge_natAbs_le hP _ (fun r ↦ hodd r c i) (fun r ↦ hbound r c i)

/-- Modular weighting followed by a single half offset, coefficient by coefficient.
Weights may be represented modulo q; only their reduced values matter. -/
theorem smudge_weighted_poly {q n P d : Nat} (hP : P % 2 = 0) {half : Int}
    (hhalf : ((2 * half : Int) : ZMod q) = 1)
    (digit : Fin (d + 1) → Fin n → Int) (hodd : ∀ i, digit 0 i % 2 = 1) :
    (∑ r : Fin (d + 1), reducePoly q n (intPoly (digit r)) *
        ((half : ExactPoly q n) * (P : ExactPoly q n) ^ r.val)) -
      reducePoly q n (intPoly (fun _ ↦ half)) =
    reducePoly q n (intPoly (fun i ↦ smudgeInt P (d + 1) (fun r ↦ digit r i))) := by
  have he (i : Fin n) :
      (∑ r : Fin (d + 1), (P : Int) ^ r.val * digit r i) % 2 = 1 := by
    have h := smudge_two_mul hP (fun r ↦ digit r i) (hodd i)
    rw [h]
    omega
  have hscalar (i : Fin n) := smudge_one_digit_value (n := n) hhalf
    (∑ r : Fin (d + 1), (P : Int) ^ r.val * digit r i) (he i)
  simp only [intPoly, map_sum, map_mul, reducePoly_intCast] at *
  simp only [Finset.sum_mul]
  rw [Finset.sum_comm, ← Finset.sum_sub_distrib]
  apply Finset.sum_congr rfl
  intro i _
  have hfactor :
      (∑ r : Fin (d + 1), (digit r i : ExactPoly q n) *
          reducePoly q n (Xe i.val) * ((half : ExactPoly q n) * (P : ExactPoly q n) ^ r.val)) -
        (half : ExactPoly q n) * reducePoly q n (Xe i.val) =
      (((∑ r : Fin (d + 1), (P : Int) ^ r.val * digit r i : Int) : ExactPoly q n) *
        half - half) * reducePoly q n (Xe i.val) := by
    push_cast
    rw [sub_mul, Finset.sum_mul, Finset.sum_mul]
    congr 1
    apply Finset.sum_congr rfl
    intro r _
    ring
  rw [hfactor, hscalar]
  rfl
end Dio2Proof
