import MxxSecurity.Distribution
import RuntimeLemmas

/-!
# Uniform matrices

Uniform residue families and matrices are invariant in distribution under coordinatewise
bijections of the residues: scaling by a unit and translation.
-/

namespace MxxSecurity

open MeasureTheory OracleComp MxxRuntime Mxx.Primitives

theorem sampleFamily_zero (law : SamplerLaw) : sampleFamily law 0 = pure Fin.elim0 := rfl

theorem sampleFamily_succ (law : SamplerLaw) (k : Nat) :
    sampleFamily law (k + 1) = (do
      let head ← sample law
      let tail ← sampleFamily law k
      pure (Fin.cons head tail)) := rfl

/-- Relabelling every coordinate of a uniform residue family by a bijection of the residues
preserves its distribution. -/
theorem sampleFamily_residue_perm {β : Type} {q : Nat} (hq : 0 < q) :
    ∀ (k : Nat) (ψ : Fin k → Int → Int), (∀ i, Set.BijOn (ψ i) (residues q) (residues q)) →
      ∀ f : (Fin k → Int) → Comp β,
        (sampleFamily (.residue q) k >>= f) =ᵈ
          (sampleFamily (.residue q) k >>= fun v ↦ f fun i ↦ ψ i (v i)) := by
  intro k
  induction k with
  | zero =>
    intro ψ _ f
    simp only [sampleFamily_zero, pure_bind]
    have : (fun i : Fin 0 ↦ ψ i (Fin.elim0 i)) = Fin.elim0 := funext fun i ↦ i.elim0
    rw [this]
  | succ k ih =>
    intro ψ hψ f
    simp only [sampleFamily_succ, bind_assoc, pure_bind]
    have hcons : ∀ head (tail : Fin k → Int),
        (fun i ↦ ψ i (Fin.cons head tail i : Int)) =
          Fin.cons (ψ 0 head) (fun j ↦ ψ j.succ (tail j)) := by
      intro head tail
      funext i
      refine Fin.cases ?_ (fun j ↦ ?_) i <;> simp
    simp only [hcons]
    refine EvalDistEq.trans (EvalDistEq.bind_right _ fun head ↦
      ih (fun j ↦ ψ j.succ) (fun j ↦ hψ j.succ) (fun tail ↦ f (Fin.cons head tail))) ?_
    exact sampleResidueValue_perm hq (hψ 0)
      (fun head ↦ sampleFamily (.residue q) k >>= fun tail ↦
        f (Fin.cons head fun j ↦ ψ j.succ (tail j)))

/-! ## Coefficients modulo `q` -/

theorem natCast_self_poly {q n : Nat} : ((q : Int) : ExactPoly q n) = 0 := by
  rw [Int.cast_natCast, ← map_natCast (algebraMap (ZMod q) (ExactPoly q n)), ZMod.natCast_self,
    map_zero]

theorem intCast_emod_poly {q n : Nat} (x : Int) :
    (((x % q : Int)) : ExactPoly q n) = (x : ExactPoly q n) := by
  rw [Int.emod_def, Int.cast_sub, Int.cast_mul, natCast_self_poly, zero_mul, sub_zero]

theorem polynomialOfCoefficients_add {q n : Nat} (a b : Fin n → Int) :
    polynomialOfCoefficients (q := q) (fun i ↦ a i + b i) =
      polynomialOfCoefficients a + polynomialOfCoefficients b := by
  unfold polynomialOfCoefficients
  rw [← Finset.sum_add_distrib]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  push_cast
  ring

theorem polynomialOfCoefficients_mul {q n : Nat} (c : Int) (a : Fin n → Int) :
    polynomialOfCoefficients (q := q) (fun i ↦ c * a i) =
      (c : ExactPoly q n) * polynomialOfCoefficients a := by
  unfold polynomialOfCoefficients
  rw [Finset.mul_sum]
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  push_cast
  ring

theorem polynomialOfCoefficients_emod {q n : Nat} (a : Fin n → Int) :
    polynomialOfCoefficients (q := q) (fun i ↦ a i % q) = polynomialOfCoefficients a := by
  unfold polynomialOfCoefficients
  refine Finset.sum_congr rfl fun i _ ↦ ?_
  rw [intCast_emod_poly]

/-! ## Bijections of the residues -/

/-- Two residues congruent modulo `q` are equal. -/
theorem residues_eq_of_dvd {q : Nat} {x y : Int} (hx : x ∈ residues q) (hy : y ∈ residues q)
    (hdiv : (q : Int) ∣ x - y) : x = y := by
  have hx' := mem_residues.mp hx
  have hy' := mem_residues.mp hy
  obtain ⟨k, hk⟩ := hdiv
  have : k = 0 := by
    by_contra hk0
    rcases lt_or_gt_of_ne hk0 with h | h <;> nlinarith
  subst this
  linarith

/-- Multiplication by a unit modulo `q`, reduced. -/
theorem bijOn_mul_emod {q : Nat} (hq : 0 < q) {c : Int} (hc : IsCoprime c (q : Int)) :
    Set.BijOn (fun x : Int ↦ c * x % q) (residues q) (residues q) := by
  obtain ⟨u, v, huv⟩ := hc
  refine ⟨fun x _ ↦ mem_residues.mpr ⟨Int.emod_nonneg _ (by omega),
    Int.emod_lt_of_pos _ (by omega)⟩, ?_, ?_⟩
  · intro x hx y hy hxy
    have hdiv : (q : Int) ∣ c * y - c * x := Int.ModEq.dvd hxy
    refine (residues_eq_of_dvd hy hx ?_).symm
    have : y - x = u * (c * y - c * x) + v * q * (y - x) := by
      linear_combination (x - y) * huv
    rw [this]
    exact dvd_add (dvd_mul_of_dvd_right hdiv _) (dvd_mul_of_dvd_left (dvd_mul_left _ _) _)
  · intro y hy
    have hy' := mem_residues.mp hy
    refine ⟨u * y % q, mem_residues.mpr ⟨Int.emod_nonneg _ (by omega),
      Int.emod_lt_of_pos _ (by omega)⟩, ?_⟩
    show c * (u * y % q) % q = y
    have hcongr : c * (u * y % q) % q = (c * u * y) % q := by
      rw [Int.mul_emod, Int.emod_emod_of_dvd _ (dvd_refl _), ← Int.mul_emod, mul_assoc]
    rw [hcongr]
    have : c * u * y = y + (-(v * y)) * q := by linear_combination y * huv
    rw [this, Int.add_mul_emod_self_right, Int.emod_eq_of_lt hy'.1 hy'.2]

/-- Translation by `e` modulo `q`. -/
theorem bijOn_add_emod {q : Nat} (hq : 0 < q) (e : Int) :
    Set.BijOn (fun x : Int ↦ (x + e) % q) (residues q) (residues q) := by
  refine ⟨fun x _ ↦ mem_residues.mpr ⟨Int.emod_nonneg _ (by omega), Int.emod_lt_of_pos _ (by omega)⟩,
    ?_, ?_⟩
  · intro x hx y hy hxy
    have hx' := mem_residues.mp hx
    have hy' := mem_residues.mp hy
    have : (q : Int) ∣ (x + e) - (y + e) := Int.ModEq.dvd hxy.symm
    have hdiv : (q : Int) ∣ x - y := by simpa using this
    obtain ⟨k, hk⟩ := hdiv
    have : k = 0 := by
      by_contra hk0
      rcases lt_or_gt_of_ne hk0 with h | h <;> nlinarith
    subst this
    linarith
  · intro y hy
    have hy' := mem_residues.mp hy
    refine ⟨(y - e) % q, mem_residues.mpr ⟨Int.emod_nonneg _ (by omega),
      Int.emod_lt_of_pos _ (by omega)⟩, ?_⟩
    show ((y - e) % q + e) % q = y
    rw [Int.add_emod, Int.emod_emod_of_dvd _ (dvd_refl _), ← Int.add_emod, sub_add_cancel,
      Int.emod_eq_of_lt hy'.1 hy'.2]

/-! ## Uniform matrices -/

/-- The matrix whose entry coefficients are `values`, in the order `sampleMatrix` draws them. -/
noncomputable def matrixOf {q n rows columns : Nat} (values : Fin (rows * columns * n) → Int) :
    ExactMatrix q n rows columns :=
  fun row column ↦ polynomialOfCoefficients fun coefficient ↦
    values (finProdFinEquiv (finProdFinEquiv (row, column), coefficient))

theorem sampleMatrix_eq (law : SamplerLaw) (q n rows columns : Nat) :
    sampleMatrix law q n rows columns =
      (do let values ← sampleFamily law (rows * columns * n); pure (matrixOf values)) := rfl

theorem sampleResidue_eq {q n rows columns : Nat} :
    (sampleResidue : Comp (ExactMatrix q n rows columns)) =
      (do let values ← sampleFamily (.residue q) (rows * columns * n); pure (matrixOf values)) :=
  rfl

theorem exactPoly_eq_coefficients {q n : Nat} (hq : 1 < q) (hn : 0 < n) (x : ExactPoly q n) :
    polynomialOfCoefficients (fun i ↦ ((x.coeff i).val : Int)) = x := by
  haveI : Fact (1 < q) := ⟨hq⟩
  refine MxxRuntime.negacyclic_ext hn fun i ↦ ?_
  rw [polynomialOfCoefficients_coeff hq hn]
  simp

theorem matrixScale_matrixOf {q n rows columns : Nat} (c : Int)
    (values : Fin (rows * columns * n) → Int) :
    MxxRuntime.matrixScale c (matrixOf (q := q) values) =
      matrixOf (fun i ↦ c * values i % q) := by
  funext row column
  simp only [MxxRuntime.matrixScale, matrixOf, polynomialOfCoefficients_emod,
    polynomialOfCoefficients_mul]

/-- The integer coefficients of `E`, indexed as `matrixOf` reads them. -/
noncomputable def coefficientsOf {q n rows columns : Nat} (E : ExactMatrix q n rows columns) :
    Fin (rows * columns * n) → Int := fun i ↦
  let position := finProdFinEquiv.symm i
  let entry := finProdFinEquiv.symm position.1
  (((E entry.1 entry.2).coeff position.2).val : Int)

theorem matrixOf_add {q n rows columns : Nat} (hq : 1 < q) (hn : 0 < n)
    (values : Fin (rows * columns * n) → Int) (E : ExactMatrix q n rows columns) :
    matrixOf (q := q) values + E = matrixOf (fun i ↦ (values i + coefficientsOf E i) % q) := by
  funext row column
  simp only [matrixOf, Matrix.add_apply, polynomialOfCoefficients_emod,
    polynomialOfCoefficients_add, coefficientsOf, Equiv.symm_apply_apply]
  rw [exactPoly_eq_coefficients hq hn]

/-- Scaling a uniform matrix by a unit preserves its distribution. -/
theorem sampleResidue_scale {β : Type} {q n rows columns : Nat} (hq : 0 < q) {c : Int}
    (hc : IsCoprime c (q : Int)) (f : ExactMatrix q n rows columns → Comp β) :
    (sampleResidue >>= fun M ↦ f (MxxRuntime.matrixScale c M)) =ᵈ (sampleResidue >>= f) := by
  simp only [sampleResidue_eq, bind_assoc, pure_bind, matrixScale_matrixOf]
  exact (sampleFamily_residue_perm hq _ (fun _ x ↦ c * x % q) (fun _ ↦ bijOn_mul_emod hq hc)
    (fun v ↦ f (matrixOf v))).symm

/-- Translating a uniform matrix preserves its distribution. -/
theorem sampleResidue_add {β : Type} {q n rows columns : Nat} (hq : 1 < q) (hn : 0 < n)
    (E : ExactMatrix q n rows columns) (f : ExactMatrix q n rows columns → Comp β) :
    (sampleResidue >>= fun M ↦ f (M + E)) =ᵈ (sampleResidue >>= f) := by
  simp only [sampleResidue_eq, bind_assoc, pure_bind, matrixOf_add hq hn]
  exact (sampleFamily_residue_perm (by omega) _ (fun i x ↦ (x + coefficientsOf E i) % q)
    (fun i ↦ bijOn_add_emod (by omega) _) (fun v ↦ f (matrixOf v))).symm

end MxxSecurity
