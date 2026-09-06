import ExponentCrt

namespace MxxExponentLut
open Mxx.Primitives
open scoped BigOperators

/-- Positional fresh digits are selected from the actual native readout table.
This supports non-Boolean tables; the finite table and radix budget can be
checked numerically, while selections come from the input/readout contract. -/
theorem refresh_readout_support {width count digitBound bound : Nat}
    (table : Fin width → Int) (selection : Fin count → Fin width)
    (weight : Fin count → Int)
    (tableSupport : ∀ entry, (table entry).natAbs ≤ digitBound)
    (budget : (∑ i : Fin count, (weight i).natAbs * digitBound) ≤ bound) :
    (∑ i : Fin count, weight i * table (selection i)).natAbs ≤ bound := by
  have sumBound (s : Finset (Fin count)) :
      (∑ i ∈ s, weight i * table (selection i)).natAbs ≤
        ∑ i ∈ s, (weight i).natAbs * digitBound := by
    induction s using Finset.induction_on with
    | empty => simp
    | @insert a s ha ih =>
      simp only [Finset.sum_insert ha]
      apply (Int.natAbs_add_le _ _).trans
      apply Nat.add_le_add _ ih
      rw [Int.natAbs_mul]
      exact Nat.mul_le_mul_left _ (tableSupport _)
  exact (sumBound Finset.univ).trans budget

/-- Lift intended fresh readout coefficients after the public half offset. -/
noncomputable def refreshFreshLift {n columns : Nat}
    (digits : Fin 1 → Fin columns → Fin n → Int) : ErrorMatrix n 1 columns :=
  fun row column ↦ ∑ i : Fin n, (digits row column i : ErrorPoly n) *
    AdjoinRoot.root (negacyclicModulus n Int) ^ i.val

theorem refreshFreshLift_coeff {n columns : Nat} (hn : 0 < n)
    (digits : Fin 1 → Fin columns → Fin n → Int) (row column i) :
    ((refreshFreshLift digits) row column).coeff i = digits row column i := by
  unfold refreshFreshLift
  rw [Negacyclic.coeff_sum]
  have hc (v : Int) : (v : ErrorPoly n) = algebraMap Int (ErrorPoly n) v := by simp
  simp_rw [hc, Negacyclic.coeff_smul, Negacyclic.coeff_root_pow hn]
  simp

theorem refreshFreshLift_bounded {n columns bound : Nat} (hn : 0 < n)
    (digits : Fin 1 → Fin columns → Fin n → Int)
    (support : ∀ row column i, (digits row column i).natAbs ≤ bound) :
    CoeffBound (refreshFreshLift digits) bound := by
  intro row column i
  rw [refreshFreshLift_coeff hn]
  exact support row column i

/-- Every actual endpoint anchor has this explicit integer lift; no sampled
anchor equation or error bound is needed to choose the CRT message lift. -/
theorem refresh_anchor_lift {q n columns : Nat} (hq : 1 < q) (hn : 0 < n)
    (anchor : ExactMatrix q n 1 columns) :
    reduceMatrix q n 1 columns (refreshFreshLift fun row column i ↦
      Int.ofNat ((anchor row column).coeff i).val) = anchor := by
  letI : Fact (1 < q) := ⟨hq⟩
  apply exactMatrix_ext
  intro row column
  rw [Negacyclic.expansion hn (reduceMatrix q n 1 columns _ row column),
    Negacyclic.expansion hn (anchor row column)]
  apply Finset.sum_congr rfl
  intro i _
  simp only [reduceMatrix]
  rw [reducePoly_coeff hq hn, refreshFreshLift_coeff hn]
  simp

/-- Lift and bound the intended fresh perturbation directly from selected
readout table entries and their positional weights. For the actual tiny bound,
the finite table/radix budget must establish `bound = 3`. -/
theorem refreshFreshLift_readout_bounded
    {n columns width count digitBound bound : Nat} (hn : 0 < n)
    (table : Fin width → Int)
    (selection : Fin 1 → Fin columns → Fin n → Fin count → Fin width)
    (weight : Fin count → Int)
    (tableSupport : ∀ entry, (table entry).natAbs ≤ digitBound)
    (budget : (∑ i : Fin count, (weight i).natAbs * digitBound) ≤ bound) :
    CoeffBound (refreshFreshLift fun row column i ↦
      ∑ digit : Fin count, weight digit * table (selection row column i digit)) bound := by
  apply refreshFreshLift_bounded hn
  intro row column i
  exact refresh_readout_support table (selection row column i) weight tableSupport budget

/-- Reconstruct the actual heterogeneous CRT levels and subtract the actual
anchor. Support is required of intended input digits, never of the output. -/
theorem refresh_crt_residual {q n columns slots bound : Nat}
    (hq : 1 < q) (hn : 0 < n)
    (source : Fin slots → Nat) (plaintext weight : Fin slots → Int)
    (levels : (slot : Fin slots) → ExactMatrix (source slot) n 1 columns)
    (anchor : ErrorMatrix n 1 columns)
    (digits : Fin 1 → Fin columns → Fin n → Int)
    (actual residual : ExactMatrix q n 1 columns)
    (runs : actual = ∑ slot : Fin slots, MxxRuntime.crtRecomposeLevel
      (plaintext slot) (weight slot) (levels slot))
    (endpoint : residual = actual - reduceMatrix q n 1 columns anchor)
    (annihilates : ∀ slot, (weight slot : ZMod q) * (plaintext slot : ZMod q) = 0)
    (partition : (∑ slot : Fin slots, (weight slot : ZMod q)) = 1)
    (rounded : ∀ slot row column (i : Fin n),
      ((plaintext slot * Int.ofNat ((levels slot row column).coeff i).val +
        Int.ofNat (source slot / 2)) / Int.ofNat (source slot)) % plaintext slot =
      ((anchor + refreshFreshLift digits) row column).coeff i % plaintext slot)
    (support : ∀ row column i, (digits row column i).natAbs ≤ bound) :
    Approx residual 0 bound := by
  obtain ⟨fresh, equation, bounded⟩ := crt_recompose_fresh_error hq hn source plaintext
    weight levels anchor (refreshFreshLift digits) actual runs annihilates partition rounded
    (refreshFreshLift_bounded hn digits support)
  refine ⟨fresh, ?_, bounded⟩
  rw [endpoint, equation]
  abel

end MxxExponentLut
