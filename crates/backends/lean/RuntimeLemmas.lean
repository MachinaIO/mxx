import RuntimeMatrixOps
import RuntimeModulusSwitch
import RuntimeHash
import Mathlib.FieldTheory.Finite.Basic

/-!
Facts about the runtime relations that application proofs compose: integer lifts of exact
polynomials, sampler bounds, coefficient-ring conversions, RNS corrections, and NTT
evaluation. Every lemma is stated for arbitrary parameters.
-/

namespace MxxRuntime

open Mxx.Primitives
open scoped BigOperators

section IntegerPolynomials

variable {n : Nat}

/-- The integer polynomial with coefficients `values`. -/
noncomputable def intPoly (values : Fin n → Int) : ErrorPoly n :=
  ∑ i : Fin n, (values i : ErrorPoly n) * AdjoinRoot.root (negacyclicModulus n Int) ^ i.val

theorem intCast_eq_algebraMap {R : Type} [CommRing R] (value : Int) :
    (value : Negacyclic n R) = algebraMap R (Negacyclic n R) (value : R) := by
  simp

theorem intPoly_coeff (hn : 0 < n) (values : Fin n → Int) (i : Fin n) :
    (intPoly values).coeff i = values i := by
  unfold intPoly
  rw [Negacyclic.coeff_sum]
  simp_rw [intCast_eq_algebraMap, Negacyclic.coeff_smul, Negacyclic.coeff_root_pow hn]
  simp

theorem negacyclic_ext {R : Type} [CommRing R] [Nontrivial R] (hn : 0 < n)
    {x y : Negacyclic n R} (h : ∀ i, x.coeff i = y.coeff i) : x = y := by
  rw [Negacyclic.expansion hn x, Negacyclic.expansion hn y]
  simp_rw [h]

theorem intPoly_coeff_self (hn : 0 < n) (x : ErrorPoly n) :
    intPoly (fun i ↦ x.coeff i) = x :=
  negacyclic_ext hn fun i ↦ intPoly_coeff hn _ i

theorem polynomialOfCoefficients_eq_reduce {q : Nat} (values : Fin n → Int) :
    polynomialOfCoefficients (q := q) values = reducePoly q n (intPoly values) := by
  unfold polynomialOfCoefficients intPoly
  rw [map_sum]
  apply Finset.sum_congr rfl
  intro i _
  have hroot : reducePoly q n (AdjoinRoot.root (negacyclicModulus n Int)) =
      AdjoinRoot.root (negacyclicModulus n (ZMod q)) := by
    simp [reducePoly]
  rw [map_mul, map_pow, hroot, map_intCast]

theorem reducePoly_coeff_eq {q : Nat} (hq : 1 < q) (hn : 0 < n) (x : ErrorPoly n) (i : Fin n) :
    (reducePoly q n x).coeff i = (x.coeff i : ZMod q) :=
  reducePoly_coeff hq hn x i

/-- Two integer polynomials reduce to the same exact polynomial exactly when their
coefficients agree modulo `q`. -/
theorem reducePoly_eq_iff {q : Nat} (hq : 1 < q) (hn : 0 < n) (x y : ErrorPoly n) :
    reducePoly q n x = reducePoly q n y ↔ ∀ i, x.coeff i ≡ y.coeff i [ZMOD q] := by
  letI : Fact (1 < q) := ⟨hq⟩
  constructor
  · intro h i
    have := congrArg (fun z : ExactPoly q n ↦ z.coeff i) h
    simp only [reducePoly_coeff hq hn] at this
    exact (ZMod.intCast_eq_intCast_iff _ _ _).mp this
  · intro h
    apply negacyclic_ext hn
    intro i
    rw [reducePoly_coeff hq hn, reducePoly_coeff hq hn]
    exact (ZMod.intCast_eq_intCast_iff _ _ _).mpr (h i)

/-- The canonical integer lift of an exact polynomial. -/
noncomputable def canonicalLift {q : Nat} (x : ExactPoly q n) : ErrorPoly n :=
  intPoly fun i ↦ ((x.coeff i).val : Int)

theorem canonicalLift_coeff {q : Nat} (hn : 0 < n) (x : ExactPoly q n) (i : Fin n) :
    (canonicalLift x).coeff i = ((x.coeff i).val : Int) :=
  intPoly_coeff hn _ i

theorem reducePoly_canonicalLift {q : Nat} (hq : 1 < q) (hn : 0 < n) (x : ExactPoly q n) :
    reducePoly q n (canonicalLift x) = x := by
  letI : Fact (1 < q) := ⟨hq⟩
  apply negacyclic_ext hn
  intro i
  rw [reducePoly_coeff hq hn, canonicalLift_coeff hn]
  simp

end IntegerPolynomials

section Samplers

variable {q n rows columns : Nat}

theorem polyNorm_le_of_coeff {x : ErrorPoly n} {bound : Nat}
    (h : ∀ i, (x.coeff i).natAbs ≤ bound) : polyNorm x ≤ bound :=
  Finset.sup_le fun i _ ↦ h i

/-- A ternary sample is the reduction of an integer witness of norm at most one. -/
theorem uniformIntervalSample_ternary {output : ExactMatrix q n rows columns}
    (h : uniformIntervalSample (-1) 1 output) :
    ∃ witness : ErrorMatrix n rows columns, output = reduceMatrix q n rows columns witness ∧
      ∀ row column, polyNorm (witness row column) ≤ 1 := by
  obtain ⟨_, witness, houtput, hbounds⟩ := h
  refine ⟨witness, houtput, fun row column ↦ polyNorm_le_of_coeff fun i ↦ ?_⟩
  have := hbounds row column i
  omega

/-- A Gaussian sample is the reduction of an integer witness within its cutoff. -/
theorem gaussianSample_bounded {sigma : Rat} {cutoff : Int}
    {output : ExactMatrix q n rows columns} (h : gaussianSample sigma cutoff output) :
    ∃ witness : ErrorMatrix n rows columns, output = reduceMatrix q n rows columns witness ∧
      ∀ row column, polyNorm (witness row column) ≤ cutoff.toNat := by
  obtain ⟨_, _, ⟨witness, houtput, hbound⟩, _⟩ := h
  exact ⟨witness, houtput, fun row column ↦ polyNorm_le_of_coeff (hbound row column)⟩

end Samplers

section Conversions

variable {n : Nat}

theorem val_intCast_emod {q : Nat} (hq : 1 < q) (value : Int) :
    (((value : ZMod q).val : Nat) : Int) = value % q := by
  letI : NeZero q := ⟨by omega⟩
  exact ZMod.val_intCast value

/-- Reducing a canonical residue to a divisor modulus is ordinary reduction. -/
theorem modulusReduce_reducePoly {q p : Nat} (hq : 1 < q) (hp : 1 < p) (hdvd : p ∣ q)
    (hn : 0 < n) (x : ErrorPoly n) :
    polynomialOfCoefficients (q := p)
        (fun i ↦ Int.ofNat ((reducePoly q n x).coeff i).val) = reducePoly p n x := by
  rw [polynomialOfCoefficients_eq_reduce, reducePoly_eq_iff hp hn]
  intro i
  rw [intPoly_coeff hn, reducePoly_coeff hq hn]
  change (((x.coeff i : ZMod q).val : Nat) : Int) ≡ x.coeff i [ZMOD p]
  rw [val_intCast_emod hq]
  exact (Int.mod_modEq _ _).of_dvd (Int.natCast_dvd_natCast.mpr hdvd)

/-- The centered representative of a small integer is the integer itself. -/
theorem centered_residue_of_small {q : Nat} (hq : 1 < q) {value : Int}
    (hsmall : 2 * value.natAbs < q) :
    (if ((value : ZMod q).val) ≤ q / 2 then (((value : ZMod q).val : Nat) : Int)
      else (((value : ZMod q).val : Nat) : Int) - q) = value := by
  have hval := val_intCast_emod hq value
  have hqpos : (0 : Int) < q := by omega
  by_cases hnonneg : 0 ≤ value
  · have hlt : value < q := by omega
    have hemod : value % q = value := Int.emod_eq_of_lt hnonneg hlt
    rw [hemod] at hval
    have hle : (value : ZMod q).val ≤ q / 2 := by omega
    rw [if_pos hle, hval]
  · have hemod : value % q = value + q := by
      rw [Int.emod_eq_iff (by omega)]
      refine ⟨by omega, by omega, ⟨1, by ring⟩⟩
    rw [hemod] at hval
    have hgt : ¬ (value : ZMod q).val ≤ q / 2 := by omega
    rw [if_neg hgt, hval]
    ring

/-- Centered rebasing preserves an integer polynomial whose coefficients are below half
the source modulus. -/
theorem centeredRebase_reducePoly {q p : Nat} (hq : 1 < q) (hn : 0 < n)
    (x : ErrorPoly n) (hsmall : ∀ i, 2 * (x.coeff i).natAbs < q) :
    polynomialOfCoefficients (q := p) (fun i ↦
      let u := ((reducePoly q n x).coeff i).val
      if u ≤ q / 2 then Int.ofNat u else Int.ofNat u - Int.ofNat q) = reducePoly p n x := by
  rw [polynomialOfCoefficients_eq_reduce]
  congr 1
  apply negacyclic_ext hn
  intro i
  rw [intPoly_coeff hn]
  simp only [reducePoly_coeff hq hn]
  exact centered_residue_of_small hq (hsmall i)

/-- The centered lift of a small integer is the integer itself. -/
theorem centeredLift_intCast {q : Nat} (hq : 1 < q) {value : Int}
    (hsmall : 2 * value.natAbs < q) : centeredLift q (value : ZMod q) = value := by
  have hval := val_intCast_emod hq value
  unfold centeredLift
  by_cases hnonneg : 0 ≤ value
  · have hemod : value % q = value := Int.emod_eq_of_lt hnonneg (by omega)
    rw [hemod] at hval
    rw [if_pos (by omega)]
    exact hval
  · have hemod : value % q = value + q := by
      rw [Int.emod_eq_iff (by omega)]
      refine ⟨by omega, by omega, ⟨1, by ring⟩⟩
    rw [hemod] at hval
    rw [if_neg (by omega), hval]
    ring

/-- The centered integer lift that `centeredRebase` re-encodes. -/
noncomputable def centeredIntLift {q : Nat} (x : ExactPoly q n) : ErrorPoly n :=
  intPoly fun i ↦
    let u := (x.coeff i).val
    if u ≤ q / 2 then Int.ofNat u else Int.ofNat u - Int.ofNat q

theorem centeredRebase_eq {q p rows columns : Nat} (input : ExactMatrix q n rows columns)
    (row : Fin rows) (column : Fin columns) :
    centeredRebase (p := p) input row column = reducePoly p n (centeredIntLift (input row column)) :=
  polynomialOfCoefficients_eq_reduce _

theorem reducePoly_centeredIntLift {q : Nat} (hq : 1 < q) (hn : 0 < n) (x : ExactPoly q n) :
    reducePoly q n (centeredIntLift x) = x := by
  letI : Fact (1 < q) := ⟨hq⟩
  apply negacyclic_ext hn
  intro i
  rw [reducePoly_coeff hq hn, centeredIntLift, intPoly_coeff hn]
  simp only
  split <;> simp

theorem polyNorm_centeredIntLift {q : Nat} (hq : 1 < q) (x : ExactPoly q n) :
    polyNorm (centeredIntLift x) ≤ q / 2 := by
  letI : NeZero q := ⟨by omega⟩
  by_cases hn : 0 < n
  · apply polyNorm_le_of_coeff
    intro i
    rw [centeredIntLift, intPoly_coeff hn]
    simp only [Int.ofNat_eq_natCast]
    have hlt := ZMod.val_lt (x.coeff i)
    split <;> omega
  · have : n = 0 := by omega
    subst this
    exact Finset.sup_le fun i _ ↦ Fin.elim0 i

end Conversions

section Rns

variable {n : Nat}

theorem rnsCentered_modEq (m : Nat) (value : Int) :
    rnsCentered m value ≡ value [ZMOD m] := by
  unfold rnsCentered
  simp only
  split
  · exact Int.mod_modEq _ _
  · have : value % m - m ≡ value % m [ZMOD m] := by
      apply Int.ModEq.symm
      exact (Int.modEq_iff_dvd).mpr ⟨-1, by ring⟩
    exact this.trans (Int.mod_modEq _ _)

theorem rnsCentered_natAbs_le {m : Nat} (hm : 0 < m) (value : Int) :
    (rnsCentered m value).natAbs ≤ m / 2 := by
  unfold rnsCentered
  simp only [Int.ofNat_eq_natCast]
  have hhalf : ((m / 2 : Nat) : Int) = (m : Int) / 2 := by omega
  rw [hhalf]
  have h0 : 0 ≤ value % (m : Int) := Int.emod_nonneg _ (by omega)
  have h1 : value % (m : Int) < m := Int.emod_lt_of_pos _ (by omega)
  split <;> omega

theorem intCast_mul_inv_of_gcd {m : Nat} (hm : 1 < m) (value : Int)
    (hgcd : Int.gcd value m = 1) : (value : ZMod m) * (value : ZMod m)⁻¹ = 1 := by
  letI : NeZero m := ⟨by omega⟩
  rw [ZMod.mul_inv_eq_gcd]
  have hval := val_intCast_emod hm value
  have : Nat.gcd (value : ZMod m).val m = 1 := by
    have h1 : Int.gcd (((value : ZMod m).val : Nat) : Int) (m : Int) = 1 := by
      rw [hval, Int.gcd_emod]
      exact hgcd
    rw [Int.gcd_natCast_natCast] at h1
    exact h1
  rw [this, Nat.cast_one]

theorem rnsInverse_mul_modEq {m : Nat} (hm : 1 < m) (value : Int)
    (hgcd : Int.gcd value m = 1) : value * rnsInverse m value ≡ 1 [ZMOD m] := by
  letI : NeZero m := ⟨by omega⟩
  rw [← ZMod.intCast_eq_intCast_iff]
  unfold rnsInverse
  push_cast
  simp only [Int.ofNat_eq_natCast, Int.cast_natCast, ZMod.natCast_zmod_val]
  exact intCast_mul_inv_of_gcd hm value hgcd

/-- With one dropped prime `r`, BGV modulus switching divides the corrected canonical lift by
`r` exactly; the correction is centered modulo `r` and makes the lift divisible by `r`. -/
theorem rnsModDown_single {q p rows columns : Nat} {basis : List Nat} {plaintext : Int}
    {input : ExactMatrix q n rows columns} {output : ExactMatrix p n rows columns} {r : Nat}
    (hrun : rnsModDownRuns basis plaintext input output)
    (hdropped : basis.filter (fun prime ↦ p % prime != 0) = [r])
    (hrp : Nat.Coprime r p) (hr : 1 < r) (hn : 0 < n) (row : Fin rows) (column : Fin columns) :
    ∃ correction : ErrorPoly n, polyNorm correction ≤ r / 2 ∧
      (∀ i, (r : Int) ∣ (canonicalLift (input row column)).coeff i +
        plaintext * correction.coeff i) ∧
      output row column = reducePoly p n (intPoly fun i ↦
        ((canonicalLift (input row column)).coeff i + plaintext * correction.coeff i) / r) := by
  obtain ⟨_, _, _, _, hp, _, _, _, hgcd, hout⟩ := hrun
  have hinv1 : rnsInverse r 1 = 1 := by
    letI : Fact (1 < r) := ⟨hr⟩
    simp [rnsInverse, ZMod.val_one]
  simp only [hdropped, List.prod_cons, List.prod_nil, mul_one, List.map_cons, List.map_nil,
    List.sum_cons, List.sum_nil, add_zero, Nat.div_self (by omega : 0 < r),
    Int.ofNat_eq_natCast, Nat.cast_one, one_mul, hinv1] at hout hgcd
  let correction : ErrorPoly n := intPoly fun i ↦
    rnsCentered r (-(((input row column).coeff i).val : Int) * rnsInverse r plaintext)
  have hcoeff (i : Fin n) : correction.coeff i =
      rnsCentered r (-(((input row column).coeff i).val : Int) * rnsInverse r plaintext) :=
    intPoly_coeff hn _ i
  have hgcd' : Int.gcd plaintext r = 1 := by simpa using hgcd
  have hdvd (i : Fin n) : (r : Int) ∣ (canonicalLift (input row column)).coeff i +
      plaintext * correction.coeff i := by
    rw [canonicalLift_coeff hn, hcoeff]
    set v : Int := (((input row column).coeff i).val : Int)
    have hc := rnsCentered_modEq r (-v * rnsInverse r plaintext)
    have hinv := rnsInverse_mul_modEq hr plaintext hgcd'
    have hmod : v + plaintext * rnsCentered r (-v * rnsInverse r plaintext) ≡ 0 [ZMOD r] :=
      calc v + plaintext * rnsCentered r (-v * rnsInverse r plaintext)
          ≡ v + plaintext * (-v * rnsInverse r plaintext) [ZMOD r] :=
            Int.ModEq.add_left _ (Int.ModEq.mul_left _ hc)
        _ = v - v * (plaintext * rnsInverse r plaintext) := by ring
        _ ≡ v - v * 1 [ZMOD r] := Int.ModEq.sub_left _ (Int.ModEq.mul_left _ hinv)
        _ = 0 := by ring
    simpa using hmod.symm.dvd
  refine ⟨correction, ?_, hdvd, ?_⟩
  · apply polyNorm_le_of_coeff
    intro i
    rw [hcoeff]
    exact rnsCentered_natAbs_le (by omega) _
  · rw [hout row column, polynomialOfCoefficients_eq_reduce, reducePoly_eq_iff (by omega) hn]
    intro i
    rw [intPoly_coeff hn, intPoly_coeff hn, ← hcoeff, ← canonicalLift_coeff hn]
    obtain ⟨k, hk⟩ := hdvd i
    rw [hk, Int.mul_ediv_cancel_left _ (by omega : (r : Int) ≠ 0)]
    have hrinv := rnsInverse_mul_modEq (by omega : 1 < p) (r : Int)
      (by simpa [Int.gcd_natCast_natCast] using hrp)
    calc (r : Int) * k * rnsInverse p r = k * ((r : Int) * rnsInverse p r) := by ring
      _ ≡ k * 1 [ZMOD p] := hrinv.mul_left _
      _ = k := by ring

/-- Chinese remaindering over three pairwise coprime moduli. -/
theorem crt_three {p1 p2 p3 : Nat} (h12 : Nat.Coprime p1 p2) (h13 : Nat.Coprime p1 p3)
    (h23 : Nat.Coprime p2 p3) {x y : Int} (h1 : x ≡ y [ZMOD p1]) (h2 : x ≡ y [ZMOD p2])
    (h3 : x ≡ y [ZMOD p3]) : x ≡ y [ZMOD (p1 * p2 * p3 : Nat)] := by
  rw [Int.modEq_iff_dvd] at *
  have h12' : IsCoprime (p1 : Int) p2 := Nat.isCoprime_iff_coprime.mpr h12
  have h3' : IsCoprime ((p1 * p2 : Nat) : Int) p3 :=
    Nat.isCoprime_iff_coprime.mpr (Nat.Coprime.mul_left h13 h23)
  have := h12'.mul_dvd h1 h2
  push_cast at h3' ⊢
  exact h3'.mul_dvd this h3

/-- One centered CRT term reconstructs `value` modulo its own modulus. -/
theorem crt_term_modEq {p m : Nat} (hp : 1 < p) (hcop : Nat.Coprime m p) (value : Int) :
    (m : Int) * rnsCentered p (value * rnsInverse p m) ≡ value [ZMOD p] := by
  have hinv := rnsInverse_mul_modEq hp (m : Int) (by simpa [Int.gcd_natCast_natCast] using hcop)
  calc (m : Int) * rnsCentered p (value * rnsInverse p m)
      ≡ m * (value * rnsInverse p m) [ZMOD p] :=
        Int.ModEq.mul_left _ (rnsCentered_modEq p _)
    _ = value * (m * rnsInverse p m) := by ring
    _ ≡ value * 1 [ZMOD p] := Int.ModEq.mul_left _ hinv
    _ = value := by ring

theorem dvd_modEq_zero {p : Nat} {m : Int} (hdvd : (p : Int) ∣ m) (x : Int) :
    m * x ≡ 0 [ZMOD p] :=
  (Int.modEq_zero_iff_dvd).mpr (dvd_mul_of_dvd_left hdvd x)

/-- The centered CRT digits of three pairwise coprime moduli reconstruct `value`
modulo their product. -/
theorem crt_three_centered {p1 p2 p3 : Nat} (h1 : 1 < p1) (h2 : 1 < p2) (h3 : 1 < p3)
    (h12 : Nat.Coprime p1 p2) (h13 : Nat.Coprime p1 p3) (h23 : Nat.Coprime p2 p3)
    (value : Int) :
    ((p2 * p3 : Nat) : Int) * rnsCentered p1 (value * rnsInverse p1 (p2 * p3 : Nat)) +
      ((p1 * p3 : Nat) : Int) * rnsCentered p2 (value * rnsInverse p2 (p1 * p3 : Nat)) +
      ((p1 * p2 : Nat) : Int) * rnsCentered p3 (value * rnsInverse p3 (p1 * p2 : Nat)) ≡
        value [ZMOD (p1 * p2 * p3 : Nat)] := by
  have d (a b : Nat) : ((a : Nat) : Int) ∣ ((a * b : Nat) : Int) := by push_cast; exact dvd_mul_right _ _
  have d' (a b : Nat) : ((b : Nat) : Int) ∣ ((a * b : Nat) : Int) := by push_cast; exact dvd_mul_left _ _
  set c1 := rnsCentered p1 (value * rnsInverse p1 (p2 * p3 : Nat))
  set c2 := rnsCentered p2 (value * rnsInverse p2 (p1 * p3 : Nat))
  set c3 := rnsCentered p3 (value * rnsInverse p3 (p1 * p2 : Nat))
  apply crt_three h12 h13 h23
  · have := (crt_term_modEq h1 (Nat.Coprime.mul_left h12.symm h13.symm) value).add
      ((dvd_modEq_zero (d p1 p3) c2).add (dvd_modEq_zero (d p1 p2) c3))
    simpa [add_assoc] using this
  · have := (dvd_modEq_zero (d p2 p3) c1).add
      ((crt_term_modEq h2 (Nat.Coprime.mul_left h12 h23.symm) value).add
        (dvd_modEq_zero (d' p1 p2) c3))
    simpa [add_assoc] using this
  · have := (dvd_modEq_zero (d' p2 p3) c1).add
      ((dvd_modEq_zero (d' p1 p3) c2).add (crt_term_modEq h3 (Nat.Coprime.mul_left h13 h23) value))
    simpa [add_assoc] using this

theorem rnsInverse_one {m : Nat} (hm : 1 < m) : rnsInverse m 1 = 1 := by
  letI : Fact (1 < m) := ⟨hm⟩
  simp [rnsInverse, ZMod.val_one]

/-- One unnormalized digit spanning a three-prime basis is its centered CRT sum. -/
theorem rnsDigit_three_whole {p1 p2 p3 : Nat} (h1 : 0 < p1) (h2 : 0 < p2) (h3 : 0 < p3)
    (value : Int) :
    rnsDigit [p1, p2, p3] 3 0 false value =
      ((p2 * p3 : Nat) : Int) * rnsCentered p1 (value * rnsInverse p1 (p2 * p3 : Nat)) +
      ((p1 * p3 : Nat) : Int) * rnsCentered p2 (value * rnsInverse p2 (p1 * p3 : Nat)) +
      ((p1 * p2 : Nat) : Int) * rnsCentered p3 (value * rnsInverse p3 (p1 * p2 : Nat)) := by
  have e1 : p1 * (p2 * p3) / p1 = p2 * p3 := Nat.mul_div_cancel_left _ h1
  have e2 : p1 * (p2 * p3) / p2 = p1 * p3 := by
    rw [show p1 * (p2 * p3) = p2 * (p1 * p3) by ring]; exact Nat.mul_div_cancel_left _ h2
  have e3 : p1 * (p2 * p3) / p3 = p1 * p2 := by
    rw [show p1 * (p2 * p3) = p3 * (p1 * p2) by ring]; exact Nat.mul_div_cancel_left _ h3
  simp only [rnsDigit, List.drop_zero, Nat.zero_mul, List.take, List.prod_cons, List.prod_nil,
    mul_one, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero,
    Bool.false_eq_true, if_false, e1, e2, e3, Int.ofNat_eq_natCast]
  ring

/-- With one prime per normalized digit, digit `j` is the centered CRT term of prime `j`. -/
theorem rnsDigit_three_single {p1 p2 p3 : Nat} (h1 : 1 < p1) (h2 : 1 < p2) (h3 : 1 < p3)
    (value : Int) :
    rnsDigit [p1, p2, p3] 1 0 true value =
        rnsCentered p1 (value * rnsInverse p1 (p2 * p3 : Nat)) ∧
    rnsDigit [p1, p2, p3] 1 1 true value =
        rnsCentered p2 (value * rnsInverse p2 (p1 * p3 : Nat)) ∧
    rnsDigit [p1, p2, p3] 1 2 true value =
        rnsCentered p3 (value * rnsInverse p3 (p1 * p2 : Nat)) := by
  have e1 : p1 * (p2 * p3) / p1 = p2 * p3 := Nat.mul_div_cancel_left _ (by omega)
  have e2 : p1 * (p2 * p3) / p2 = p1 * p3 := by
    rw [show p1 * (p2 * p3) = p2 * (p1 * p3) by ring]; exact Nat.mul_div_cancel_left _ (by omega)
  have e3 : p1 * (p2 * p3) / p3 = p1 * p2 := by
    rw [show p1 * (p2 * p3) = p3 * (p1 * p2) by ring]; exact Nat.mul_div_cancel_left _ (by omega)
  refine ⟨?_, ?_, ?_⟩ <;>
  simp only [rnsDigit, List.drop, List.take, List.prod_cons,
    List.prod_nil, mul_one, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero,
    if_true, ← Int.natCast_div, e1, e2, e3, Nat.div_self (by omega : 0 < p1), Nat.div_self (by omega : 0 < p2),
    Nat.div_self (by omega : 0 < p3), Int.ofNat_eq_natCast, Nat.cast_one, one_mul,
    rnsInverse_one h1, rnsInverse_one h2, rnsInverse_one h3]

theorem intCast_mul_coeff {n : Nat} (c : Int) (x : ErrorPoly n) (i : Fin n) :
    ((c : ErrorPoly n) * x).coeff i = c * x.coeff i := by
  rw [intCast_eq_algebraMap, Negacyclic.coeff_smul]
  simp

/-- `rnsModDown_single` as an exact integer equation `r * K = lift + plaintext * correction`
whose quotient `K` the output reduces. -/
theorem rnsModDown_single_exact {q p rows columns : Nat} {basis : List Nat} {plaintext : Int}
    {input : ExactMatrix q n rows columns} {output : ExactMatrix p n rows columns} {r : Nat}
    (hrun : rnsModDownRuns basis plaintext input output)
    (hdropped : basis.filter (fun prime ↦ p % prime != 0) = [r])
    (hrp : Nat.Coprime r p) (hr : 1 < r) (hn : 0 < n) (row : Fin rows) (column : Fin columns) :
    ∃ correction quotient : ErrorPoly n, polyNorm correction ≤ r / 2 ∧
      ((r : Int) : ErrorPoly n) * quotient =
        canonicalLift (input row column) + (plaintext : ErrorPoly n) * correction ∧
      output row column = reducePoly p n quotient := by
  obtain ⟨correction, hbound, hdvd, hout⟩ := rnsModDown_single hrun hdropped hrp hr hn row column
  refine ⟨correction, _, hbound, ?_, hout⟩
  apply negacyclic_ext hn
  intro i
  rw [intCast_mul_coeff, intPoly_coeff hn, Negacyclic.coeff_add, intCast_mul_coeff]
  exact Int.mul_ediv_cancel' (hdvd i)

end Rns

section Ntt

variable {n : Nat}

/-- A primitive `2n`-th root of unity modulo a prime has `n`-th power `-1`. -/
theorem nttPrimitiveRoot_pow_n {t : Nat} [Fact t.Prime] (hn : 0 < n) {root : Nat}
    (h : nttPrimitiveRoot t n root) : ((root : ZMod t)) ^ n = -1 := by
  haveI : IsDomain (ZMod t) := inferInstance
  obtain ⟨_, _, hfull, hprimitive⟩ := h
  have hsq : (((root : ZMod t)) ^ n) ^ 2 = 1 := by
    rw [← pow_mul, mul_comm, ← Nat.cast_pow]
    rw [← ZMod.natCast_mod, hfull]
    simp
  have hne : ((root : ZMod t)) ^ n ≠ 1 := by
    intro heq
    apply hprimitive n hn (by omega)
    have : ((root ^ n : Nat) : ZMod t) = ((1 : Nat) : ZMod t) := by push_cast; exact heq
    rw [ZMod.natCast_eq_natCast_iff'] at this
    rw [this]
    exact Nat.mod_eq_of_lt (Fact.out : t.Prime).one_lt
  rcases (sq_eq_one_iff (R := ZMod t)).mp hsq with h1 | h1
  · exact absurd h1 hne
  · exact h1

/-- Evaluation of a negacyclic polynomial at the odd power `root^(2 index + 1)`. -/
noncomputable def oddPowerSum {t : Nat} (root : ZMod t) (index : Nat) (x : ExactPoly t n) :
    ZMod t :=
  ∑ c : Fin n, x.coeff c * root ^ ((2 * index + 1) * c.val)

/-- The odd-power evaluation as a ring map out of the negacyclic quotient. -/
noncomputable def oddPowerEval {t : Nat} (root : ZMod t) (index : Nat) (hroot : root ^ n = -1) :
    ExactPoly t n →+* ZMod t :=
  AdjoinRoot.lift (RingHom.id (ZMod t)) (root ^ (2 * index + 1)) (by
    simp only [negacyclicModulus, Polynomial.eval₂_add, Polynomial.eval₂_X_pow,
      Polynomial.eval₂_C, RingHom.id_apply]
    rw [← pow_mul, mul_comm, pow_mul, hroot, pow_succ, pow_mul]
    simp)

theorem oddPowerEval_eq_sum {t : Nat} [Fact (1 < t)] (hn : 0 < n) (root : ZMod t)
    (index : Nat) (hroot : root ^ n = -1) (x : ExactPoly t n) :
    oddPowerEval root index hroot x = oddPowerSum root index x := by
  conv_lhs => rw [Negacyclic.expansion hn x]
  rw [map_sum]
  apply Finset.sum_congr rfl
  intro c _
  rw [map_mul, map_pow]
  simp only [oddPowerEval, AdjoinRoot.lift_root]
  rw [show algebraMap (ZMod t) (ExactPoly t n) (x.coeff c) =
      AdjoinRoot.of (negacyclicModulus n (ZMod t)) (x.coeff c) from rfl, AdjoinRoot.lift_of]
  simp [pow_mul]

/-- Odd-power evaluation is multiplicative on the negacyclic quotient. -/
theorem oddPowerSum_mul {t : Nat} [Fact (1 < t)] (hn : 0 < n) {root : ZMod t}
    (hroot : root ^ n = -1) (index : Nat) (x y : ExactPoly t n) :
    oddPowerSum root index (x * y) = oddPowerSum root index x * oddPowerSum root index y := by
  rw [← oddPowerEval_eq_sum hn root index hroot, ← oddPowerEval_eq_sum hn root index hroot,
    ← oddPowerEval_eq_sum hn root index hroot, map_mul]

theorem natCast_eq_of_mod_eq {t root r : Nat} (hroot : root % t = r) :
    (root : ZMod t) = (r : ZMod t) := by
  rw [← hroot, ZMod.natCast_mod]

/-- An NTT relation over a prime modulus evaluates at the native root's odd powers. -/
theorem polynomialNttRuns_eval {t : Nat} [ht : Fact t.Prime] {inverse : Bool}
    {input output : Fin n → Int} (h : polynomialNttRuns t inverse input output) :
    nttPrimitiveRoot t n (nativeNttRoot t n) ∧ (∀ i, 0 ≤ output i ∧ output i < t) ∧
    ∃ bits : Nat, n = 2 ^ bits ∧ ∀ index : Fin n, ∃ slot : Fin n,
      slot.val = nttBitReverse bits index.val ∧
      ((if inverse then input else output) slot : ZMod t) =
        ∑ c : Fin n, (((if inverse then output else input) c : Int) : ZMod t) *
          (nativeNttRoot t n : ZMod t) ^ ((2 * index.val + 1) * c.val) := by
  obtain ⟨_, _, bits, root, hbits, hlt, hlimbs, hcanonical, hevaluation⟩ := h
  obtain ⟨_, hprimitive, hnative⟩ := hlimbs t ht.out (dvd_refl t)
  refine ⟨hnative ▸ hprimitive, hcanonical, bits, hbits, fun index ↦ ?_⟩
  have hroot : (root : ZMod t) = (nativeNttRoot t n : ZMod t) := natCast_eq_of_mod_eq hnative
  cases inverse <;> simp only [Bool.false_eq_true, if_false, if_true] at hevaluation ⊢ <;>
  · obtain ⟨slot, hslot, heq⟩ := hevaluation index
    refine ⟨slot, hslot, ?_⟩
    have := (ZMod.intCast_eq_intCast_iff' _ _ t).mpr heq
    rw [this]
    push_cast
    rw [hroot]

/-- Bit reversal by repeated halving, which the kernel evaluates directly. -/
def bitReverseFast : Nat → Nat → Nat → Nat
  | 0, _, acc => acc
  | bits + 1, index, acc => bitReverseFast bits (index / 2) (2 * acc + index % 2)

theorem nttBitReverse_succ (bits index : Nat) :
    nttBitReverse (bits + 1) index = index % 2 * 2 ^ bits + nttBitReverse bits (index / 2) := by
  unfold nttBitReverse
  rw [Finset.sum_range_succ']
  simp only [pow_zero, Nat.div_one, Nat.add_sub_cancel, Nat.sub_zero]
  rw [add_comm]
  congr 1
  apply Finset.sum_congr rfl
  intro bit hbit
  rw [Finset.mem_range] at hbit
  rw [pow_succ, ← Nat.div_div_eq_div_mul, show bits - (bit + 1) = bits - 1 - bit by omega,
    Nat.div_div_eq_div_mul, Nat.div_div_eq_div_mul, mul_comm (2 ^ bit) 2]

theorem bitReverseFast_eq (bits index acc : Nat) :
    bitReverseFast bits index acc = acc * 2 ^ bits + nttBitReverse bits index := by
  induction bits generalizing index acc with
  | zero => simp [bitReverseFast, nttBitReverse]
  | succ bits ih =>
    rw [bitReverseFast, ih, nttBitReverse_succ, pow_succ]
    ring

end Ntt

section Tables

variable {n : Nat}

theorem packedPolynomial_coeff {t : Nat} (ht : 1 < t) (hn : 0 < n) (width count table : Nat)
    (i : Fin n) :
    ((packedPolynomial (q := t) (n := n) width count table) 0 0).coeff i =
      ((if i.val < count then (packedEntry width table i.val : Int) else 0 : Int) : ZMod t) := by
  unfold packedPolynomial
  exact polynomialOfCoefficients_coeff ht hn _ i

/-- Exported canonical coefficients of a packed table below the modulus are its fields. -/
theorem polynomialValues_packed {t : Nat} (ht : 1 < t) (hn : 0 < n) {width table : Nat}
    {output : Fin n → Int}
    (hrun : polynomialValues false (packedPolynomial (q := t) (n := n) width n table) output)
    (hsmall : ∀ i : Fin n, packedEntry width table i.val < t) (i : Fin n) :
    output i = packedEntry width table i.val := by
  letI : Fact (1 < t) := ⟨ht⟩
  simp only [polynomialValues, Bool.false_eq_true, if_false] at hrun
  rw [hrun i, packedPolynomial_coeff ht hn, if_pos i.isLt]
  push_cast
  rw [ZMod.val_natCast, Nat.mod_eq_of_lt (hsmall i)]

end Tables

section Shapes

variable {q n c : Nat}

theorem concatRows_one_one {left right : ExactMatrix q n 1 c} {output : ExactMatrix q n (1 + 1) c}
    (h : concatRows left right output) (column : Fin c) :
    output 0 column = left 0 column ∧ output 1 column = right 0 column :=
  ⟨by rw [h]; rfl, by rw [h]; rfl⟩

theorem concatRows_two_one {left : ExactMatrix q n 2 c} {right : ExactMatrix q n 1 c}
    {output : ExactMatrix q n (2 + 1) c} (h : concatRows left right output) (column : Fin c) :
    output 0 column = left 0 column ∧ output 1 column = left 1 column ∧
      output 2 column = right 0 column :=
  ⟨by rw [h]; rfl, by rw [h]; rfl, by rw [h]; rfl⟩

theorem concatColumns_one_one {left right : ExactMatrix q n 1 1}
    {output : ExactMatrix q n 1 (1 + 1)} (h : concatColumns left right output) :
    output 0 0 = left 0 0 ∧ output 0 1 = right 0 0 :=
  ⟨by rw [h]; rfl, by rw [h]; rfl⟩

theorem concatColumns_two_one {left : ExactMatrix q n 1 2} {right : ExactMatrix q n 1 1}
    {output : ExactMatrix q n 1 (2 + 1)} (h : concatColumns left right output) :
    output 0 0 = left 0 0 ∧ output 0 1 = left 0 1 ∧ output 0 2 = right 0 0 :=
  ⟨by rw [h]; rfl, by rw [h]; rfl, by rw [h]; rfl⟩

/-- One selected entry of a matrix slice. -/
theorem sliceMatrix_entry {rows columns : Nat} {input : ExactMatrix q n rows columns}
    {output : ExactMatrix q n 1 1} {rowStart rowEnd columnStart columnEnd : Int}
    (h : sliceMatrix input rowStart rowEnd columnStart columnEnd output)
    (hrow : rowStart.toNat < rows) (hcolumn : columnStart.toNat < columns) :
    output 0 0 = input ⟨rowStart.toNat, hrow⟩ ⟨columnStart.toNat, hcolumn⟩ := by
  obtain ⟨_, _, _, _, _, _, _, _, hout⟩ := h
  exact hout 0 0 hrow hcolumn

theorem tensorRuns_two {left right : ExactMatrix q n 2 1} {output : ExactMatrix q n (2 * 2) (1 * 1)}
    (h : tensorRuns left right output) :
    output 0 0 = left 0 0 * right 0 0 ∧ output 1 0 = left 0 0 * right 1 0 ∧
      output 2 0 = left 1 0 * right 0 0 ∧ output 3 0 = left 1 0 * right 1 0 :=
  ⟨h 0 0 0 0, h 0 1 0 0, h 1 0 0 0, h 1 1 0 0⟩

theorem concatColumns_castAdd {rows l r : Nat} {left : ExactMatrix q n rows l}
    {right : ExactMatrix q n rows r} {output : ExactMatrix q n rows (l + r)}
    (h : concatColumns left right output) (row : Fin rows) (column : Fin l) :
    output row (Fin.castAdd r column) = left row column := by
  rw [h]
  simp

theorem concatColumns_natAdd {rows l r : Nat} {left : ExactMatrix q n rows l}
    {right : ExactMatrix q n rows r} {output : ExactMatrix q n rows (l + r)}
    (h : concatColumns left right output) (row : Fin rows) (column : Fin r) :
    output row (Fin.natAdd l column) = right row column := by
  rw [h]
  simp

end Shapes

section IntegerArithmetic

variable {n : Nat}

/-- Equal reductions differ by a multiple of the modulus. -/
theorem exists_of_reducePoly_eq {q : Nat} (hq : 1 < q) (hn : 0 < n) {x y : ErrorPoly n}
    (h : reducePoly q n x = reducePoly q n y) :
    ∃ z : ErrorPoly n, x = y + ((q : Int) : ErrorPoly n) * z := by
  have hmod := (reducePoly_eq_iff hq hn x y).mp h
  refine ⟨intPoly fun i ↦ (x.coeff i - y.coeff i) / q, negacyclic_ext hn fun i ↦ ?_⟩
  rw [Negacyclic.coeff_add, intCast_mul_coeff, intPoly_coeff hn]
  have hdvd : (q : Int) ∣ x.coeff i - y.coeff i := (hmod i).symm.dvd
  rw [Int.mul_ediv_cancel' hdvd]
  ring

theorem reducePoly_intCast (q : Nat) (c : Int) :
    reducePoly q n (c : ErrorPoly n) = (c : ExactPoly q n) := map_intCast _ c

theorem reducePoly_modulus_mul (q : Nat) (z : ErrorPoly n) :
    reducePoly q n (((q : Int) : ErrorPoly n) * z) = 0 := by
  rw [map_mul, reducePoly_intCast, intCast_eq_algebraMap]
  simp

/-- A nonzero integer scalar cancels from an integer polynomial equation. -/
theorem intCast_mul_cancel (hn : 0 < n) {c : Int} (hc : c ≠ 0) {x y : ErrorPoly n}
    (h : (c : ErrorPoly n) * x = (c : ErrorPoly n) * y) : x = y := by
  apply negacyclic_ext hn
  intro i
  have := congrArg (fun z : ErrorPoly n ↦ z.coeff i) h
  simp only [intCast_mul_coeff] at this
  exact mul_left_cancel₀ hc this

/-- If `a * x` is a multiple of `b` with `a` coprime to `b`, then `x` is. -/
theorem exists_of_coprime_mul (hn : 0 < n) {a b : Int} (hab : Int.gcd a b = 1) {x y : ErrorPoly n}
    (h : (a : ErrorPoly n) * x = (b : ErrorPoly n) * y) :
    ∃ z : ErrorPoly n, x = (b : ErrorPoly n) * z := by
  have hdvd (i : Fin n) : b ∣ x.coeff i := by
    have := congrArg (fun w : ErrorPoly n ↦ w.coeff i) h
    simp only [intCast_mul_coeff] at this
    have hb : b ∣ a * x.coeff i := ⟨y.coeff i, this⟩
    exact Int.dvd_of_dvd_mul_right_of_gcd_one hb (by rw [Int.gcd_comm]; exact hab)
  refine ⟨intPoly fun i ↦ x.coeff i / b, negacyclic_ext hn fun i ↦ ?_⟩
  rw [intCast_mul_coeff, intPoly_coeff hn, Int.mul_ediv_cancel' (hdvd i)]

/-- The centered lift of the reduction of a small integer polynomial is that polynomial. -/
theorem centeredIntLift_reducePoly {q : Nat} (hq : 1 < q) (hn : 0 < n) (x : ErrorPoly n)
    (hsmall : ∀ i, 2 * (x.coeff i).natAbs < q) : centeredIntLift (reducePoly q n x) = x := by
  apply negacyclic_ext hn
  intro i
  rw [centeredIntLift, intPoly_coeff hn]
  simp only [reducePoly_coeff hq hn]
  exact centered_residue_of_small hq (hsmall i)

theorem natAbs_coeff_le_of_polyNorm {x : ErrorPoly n} {bound : Nat} (h : polyNorm x ≤ bound)
    (i : Fin n) : (x.coeff i).natAbs ≤ bound :=
  (coeff_natAbs_le_polyNorm x i).trans h

theorem polyNorm_intCast_mul_le (c : Int) (x : ErrorPoly n) :
    polyNorm ((c : ErrorPoly n) * x) ≤ c.natAbs * polyNorm x := by
  rw [intCast_eq_algebraMap]
  exact polyNorm_int_smul_le c x

end IntegerArithmetic

section Exhaustive

/-- Checks `test` below `bound` by structural recursion, which the kernel evaluates. -/
def checkBelow (test : Nat → Bool) : Nat → Bool
  | 0 => true
  | k + 1 => test k && checkBelow test k

theorem checkBelow_spec {test : Nat → Bool} {bound : Nat} (h : checkBelow test bound = true) :
    ∀ k < bound, test k = true := by
  induction bound with
  | zero => intro k hk; omega
  | succ bound ih =>
    simp only [checkBelow, Bool.and_eq_true] at h
    intro k hk
    rcases Nat.lt_succ_iff_lt_or_eq.mp hk with hlt | heq
    · exact ih h.2 k hlt
    · exact heq ▸ h.1

end Exhaustive

section IntegerFamilies

/-- Row `position` of a row-major integer matrix family times a vector. -/
theorem intMatrixVectorProduct_rows {entries inner outer : Nat} (hentries : entries = outer * inner)
    (matrix : Fin entries → Int) (vector : Fin inner → Int) (position : Fin outer) :
    intMatrixVectorProduct false matrix vector position =
      ∑ term : Fin inner, matrix ⟨position.val * inner + term.val, by
        subst hentries
        have := position.isLt; have := term.isLt; nlinarith⟩ * vector term := by
  unfold intMatrixVectorProduct
  apply Finset.sum_congr rfl
  intro term _
  simp only [Bool.false_eq_true, if_false]
  rw [dif_pos (by subst hentries; have := position.isLt; have := term.isLt; nlinarith)]

/-- Column `position` of a row-major integer matrix family, weighted by a vector. -/
theorem intMatrixVectorProduct_columns {entries inner outer : Nat}
    (hentries : entries = inner * outer) (matrix : Fin entries → Int) (vector : Fin inner → Int)
    (position : Fin outer) :
    intMatrixVectorProduct true matrix vector position =
      ∑ term : Fin inner, matrix ⟨term.val * outer + position.val, by
        subst hentries
        have := position.isLt; have := term.isLt; nlinarith⟩ * vector term := by
  unfold intMatrixVectorProduct
  apply Finset.sum_congr rfl
  intro term _
  simp only [if_true]
  rw [dif_pos (by subst hentries; have := position.isLt; have := term.isLt; nlinarith)]

theorem hashIntFamily_range {count : Nat} {model : HashModel} {modulus : Int} {tagPrefix : Blob}
    {components : List HashTagComponent} {key : ByteArray} {output : Fin count → Int}
    (h : hashIntFamily model modulus tagPrefix components key output) (index : Fin count) :
    0 ≤ output index ∧ output index < modulus := by
  obtain ⟨_, _, ⟨bits, hbits, hmodulus⟩, hout⟩ := h
  rw [hout]
  dsimp only
  have hpos : 0 < modulus.toNat := by rw [hmodulus]; simp
  refine ⟨by positivity, ?_⟩
  have := Nat.mod_lt (model.integers count modulus.toNat key
    (completeHashTag tagPrefix components) index) hpos
  have hm : (modulus.toNat : Int) = modulus := Int.toNat_of_nonneg (by rw [hmodulus]; positivity)
  omega

/-- A hash family is a function of its model, modulus, tag and key. -/
theorem hashIntFamily_functional {count : Nat} {model : HashModel} {modulus : Int}
    {tagPrefix : Blob} {components : List HashTagComponent} {key : ByteArray}
    {output output' : Fin count → Int}
    (h : hashIntFamily model modulus tagPrefix components key output)
    (h' : hashIntFamily model modulus tagPrefix components key output') : output = output' :=
  h.2.2.2.trans h'.2.2.2.symm

/-- The centered value that `select` computes from a canonical residue of a small integer. -/
theorem centered_select_of_small {q : Nat} (hq : 1 < q) {value : Int}
    (hsmall : 2 * value.natAbs < q) :
    (if 2 * (value % q) ≤ q then value % q else value % q - q) = value := by
  have hqpos : (0 : Int) < q := by omega
  by_cases hnonneg : 0 ≤ value
  · rw [Int.emod_eq_of_lt hnonneg (by omega), if_pos (by omega)]
  · have hemod : value % q = value + q := by
      rw [Int.emod_eq_iff (by omega)]
      refine ⟨by omega, by omega, ⟨1, by ring⟩⟩
    rw [hemod, if_neg (by omega)]
    ring

end IntegerFamilies

section Gadgets

theorem castExactMatrix_entry {a b n r c r' c' : Nat} (h : a = b) (M : ExactMatrix a n r c)
    (M' : ExactMatrix a n r' c') {i : Fin r} {j : Fin c} {i' : Fin r'} {j' : Fin c'}
    (hentry : M i j = M' i' j') : castExactMatrix h M i j = castExactMatrix h M' i' j' := by
  subst h
  exact hentry

theorem castExactMatrix_zero {a b n r c : Nat} (h : a = b) (M : ExactMatrix a n r c)
    {i : Fin r} {j : Fin c} (hentry : M i j = 0) : castExactMatrix h M i j = 0 := by
  subst h
  exact hentry

/-- A limb is determined by its flattened position. -/
theorem regularLimb_ext {q : Nat} {layout : RegularLayout q} {x y : RegularLimb layout}
    (h : x.1.val * layout.digitsPerTower + x.2.val = y.1.val * layout.digitsPerTower + y.2.val) :
    x = y := by
  obtain ⟨xt, xd⟩ := x
  obtain ⟨yt, yd⟩ := y
  simp only at h
  have hxd := xd.isLt
  have hyd := yd.isLt
  have ht : xt.val = yt.val := by
    have := congrArg (· / layout.digitsPerTower) h
    simp only [Nat.mul_comm _ layout.digitsPerTower] at this
    rwa [Nat.mul_add_div (by omega), Nat.mul_add_div (by omega), Nat.div_eq_of_lt hxd,
      Nat.div_eq_of_lt hyd, add_zero, add_zero] at this
  have htt : xt = yt := Fin.ext ht
  subst htt
  have hd : xd = yd := Fin.ext (by omega)
  subst hd
  rfl

/-- The inverse index equivalence splits a column into its row and flattened limb. -/
theorem regularIndexEquiv_symm {q rows : Nat} (layout : RegularLayout q)
    (index : Fin (rows * (layout.crtModuli.length * layout.digitsPerTower))) :
    ((regularIndexEquiv layout rows).symm index).1.val =
        index.val / (layout.crtModuli.length * layout.digitsPerTower) ∧
      ((regularIndexEquiv layout rows).symm index).2.1.val * layout.digitsPerTower +
        ((regularIndexEquiv layout rows).symm index).2.2.val =
        index.val % (layout.crtModuli.length * layout.digitsPerTower) := by
  set p := (regularIndexEquiv layout rows).symm index
  have hval := regularIndexEquiv_val layout p.1 p.2.1 p.2.2
  have hp : regularIndexEquiv layout rows (p.1, ⟨p.2.1, p.2.2⟩) = index := by
    simp [p]
  rw [hp] at hval
  have ht := p.2.1.isLt
  have hd := p.2.2.isLt
  have hlimb : p.2.1.val * layout.digitsPerTower + p.2.2.val <
      layout.crtModuli.length * layout.digitsPerTower := by
    calc p.2.1.val * layout.digitsPerTower + p.2.2.val
        < p.2.1.val * layout.digitsPerTower + layout.digitsPerTower := by omega
      _ = (p.2.1.val + 1) * layout.digitsPerTower := by ring
      _ ≤ layout.crtModuli.length * layout.digitsPerTower := Nat.mul_le_mul_right _ ht
  constructor
  · rw [hval, add_assoc, Nat.mul_comm p.1.val, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hlimb,
      add_zero]
  · rw [hval, add_assoc, Nat.mul_comm p.1.val, Nat.mul_add_mod, Nat.mod_eq_of_lt hlimb]

theorem regularColumnIndex_val_of_exact {q rows : Nat} (layout : RegularLayout q)
    (hexact : layout.droppedModuli = 0) (index : Fin (rows * layout.digitCount)) :
    (regularColumnIndex layout index).val = index.val := by
  simp [regularColumnIndex, RegularLayout.digitCount, RegularLayout.retainedTowers, hexact,
    finProdFinEquiv, Nat.mod_add_div]

theorem digitCount_of_exact {q : Nat} (layout : RegularLayout q)
    (hexact : layout.droppedModuli = 0) :
    layout.digitCount = layout.crtModuli.length * layout.digitsPerTower := by
  simp [RegularLayout.digitCount, RegularLayout.retainedTowers, hexact]

/-- Without dropped limbs, the two-row gadget matrix is block diagonal in the one-row gadget. -/
theorem regularGadgetMatrix_two_rows {q n : Nat} (layout : RegularLayout q)
    (hexact : layout.droppedModuli = 0) (row : Fin 2) (column : Fin (2 * layout.digitCount)) :
    regularGadgetMatrix (n := n) (rows := 2) layout row column =
      if column.val / layout.digitCount = row.val then
        regularGadgetMatrix (n := n) (rows := 1) layout 0
          ⟨column.val % layout.digitCount, by
            have := column.isLt
            have hpos : 0 < layout.digitCount := by omega
            simpa using Nat.mod_lt _ hpos⟩
      else 0 := by
  have hcount := digitCount_of_exact layout hexact
  have hpos : 0 < layout.digitCount := by
    have := column.isLt
    omega
  set c' : Fin (1 * layout.digitCount) := ⟨column.val % layout.digitCount, by
    simpa using Nat.mod_lt _ hpos⟩
  have hc2 := regularColumnIndex_val_of_exact layout hexact column
  have hc1 := regularColumnIndex_val_of_exact layout hexact c'
  obtain ⟨hrow2, hlimb2⟩ := regularIndexEquiv_symm layout (regularColumnIndex layout column)
  obtain ⟨hrow1, hlimb1⟩ := regularIndexEquiv_symm (rows := 1) layout (regularColumnIndex layout c')
  rw [hc2] at hrow2 hlimb2
  rw [hc1] at hlimb1
  replace hrow2 := hrow2.trans (congrArg (column.val / ·) hcount.symm)
  replace hlimb2 := hlimb2.trans (congrArg (column.val % ·) hcount.symm)
  replace hlimb1 := hlimb1.trans (congrArg (c'.val % ·) hcount.symm)
  simp only at hrow2 hlimb2 hlimb1
  unfold regularGadgetMatrix fullGadgetMatrix
  split
  · rename_i heq
    apply castExactMatrix_entry
    simp only [regularGadgetUnflattened]
    have hr : row = ((regularIndexEquiv layout 2).symm (regularColumnIndex layout column)).1 :=
      Fin.ext (by rw [hrow2]; exact heq.symm)
    have h0 : (0 : Fin 1) = ((regularIndexEquiv layout 1).symm (regularColumnIndex layout c')).1 :=
      Subsingleton.elim _ _
    rw [if_pos hr, if_pos h0]
    congr 2
    apply regularLimb_ext
    rw [hlimb2, hlimb1]
    simp [c']
  · rename_i hne
    apply castExactMatrix_zero
    simp only [regularGadgetUnflattened]
    rw [if_neg]
    intro hr
    apply hne
    rw [← hrow2, hr]

theorem castMatrixColumns_apply {q n rows columns columns' : Nat} (h : columns = columns')
    (value : ExactMatrix q n rows columns) (i : Fin rows) (j : Fin columns') :
    castMatrixColumns h value i j = value i (Fin.cast h.symm j) := by
  subst h
  rfl

theorem castMatrixRows_apply {q n rows rows' columns : Nat} (h : rows = rows')
    (value : ExactMatrix q n rows columns) (i : Fin rows') (j : Fin columns) :
    castMatrixRows h value i j = value (Fin.cast h.symm i) j := by
  subst h
  rfl

theorem preimageWithin_castMatrixRows {q n rows rows' columns : Nat} (h : rows = rows')
    {value : ExactMatrix q n rows columns} {bound : Nat} (hvalue : PreimageWithin value bound) :
    PreimageWithin (castMatrixRows h value) bound := by
  subst h
  exact hvalue

end Gadgets

section Negacyclic

variable {n : Nat}

theorem expansion_scaledBasis (hn : 0 < n) (x : ErrorPoly n) :
    x = ∑ j : Fin n, scaledBasis (x.coeff j) j :=
  Negacyclic.expansion hn x

theorem root_eq_scaledBasis (hn : 1 < n) :
    AdjoinRoot.root (negacyclicModulus n Int) = scaledBasis 1 ⟨1, hn⟩ := by
  rw [scaledBasis, map_one, one_mul, pow_one]

/-- One negacyclic shift permutes coefficients up to sign. -/
theorem polyNorm_root_mul_le (hn : 1 < n) (x : ErrorPoly n) :
    polyNorm (AdjoinRoot.root (negacyclicModulus n Int) * x) ≤ polyNorm x := by
  apply polyNorm_le_of_coeff
  intro k
  have hn0 : 0 < n := by omega
  conv_lhs => rw [root_eq_scaledBasis hn, expansion_scaledBasis hn0 x, Finset.mul_sum,
    Negacyclic.coeff_sum]
  simp_rw [coeff_scaled_basis_mul hn0]
  rw [Finset.sum_eq_single (matchingIndex ⟨1, hn⟩ k)]
  · split
    · split
      · simpa using coeff_natAbs_le_polyNorm x _
      · simpa using coeff_natAbs_le_polyNorm x _
    · simp
  · intro j _ hj
    rw [if_neg]
    intro h
    exact hj ((matchingIndex_unique ⟨1, hn⟩ j k).mp h)
  · simp

theorem polyNorm_root_pow_mul_le (hn : 1 < n) (k : Nat) (x : ErrorPoly n) :
    polyNorm (AdjoinRoot.root (negacyclicModulus n Int) ^ k * x) ≤ polyNorm x := by
  induction k with
  | zero => simp
  | succ k ih =>
    rw [pow_succ, mul_comm (_ ^ k), mul_assoc]
    exact (polyNorm_root_mul_le hn _).trans ih

/-- The constant coefficient of a negacyclic product. -/
theorem coeff_zero_mul (hn : 0 < n) (x y : ErrorPoly n) :
    (x * y).coeff ⟨0, hn⟩ = ∑ j : Fin n,
      (if j.val = 0 then 1 else -1) * x.coeff ⟨(n - j.val) % n, Nat.mod_lt _ hn⟩ * y.coeff j := by
  conv_lhs => rw [expansion_scaledBasis hn x, expansion_scaledBasis hn y]
  rw [Finset.sum_mul, Negacyclic.coeff_sum]
  simp_rw [Finset.mul_sum, Negacyclic.coeff_sum, coeff_scaled_basis_mul hn]
  rw [Finset.sum_comm]
  apply Finset.sum_congr rfl
  intro j _
  rw [Finset.sum_eq_single ⟨(n - j.val) % n, Nat.mod_lt _ hn⟩]
  · have hj := j.isLt
    by_cases h0 : j.val = 0
    · simp only [h0, add_zero]
      simp [Nat.mod_self, hn]
    · have hsub : (n - j.val) % n = n - j.val := Nat.mod_eq_of_lt (by omega)
      simp only [hsub, if_neg h0]
      rw [if_pos (by rw [show n - j.val + j.val = n by omega, Nat.mod_self]),
        if_neg (by omega)]
      ring
  · intro i _ hi
    rw [if_neg]
    intro hmod
    apply hi
    apply Fin.ext
    have hi' := i.isLt
    have hj := j.isLt
    by_cases h0 : j.val = 0
    · simp only [h0, Nat.sub_zero, Nat.mod_self]
      simp only [h0, add_zero] at hmod
      rwa [Nat.mod_eq_of_lt hi'] at hmod
    · have hsub : (n - j.val) % n = n - j.val := Nat.mod_eq_of_lt (by omega)
      simp only [hsub]
      have : i.val + j.val = n := by
        rcases Nat.lt_or_ge (i.val + j.val) n with hl | hl
        · rw [Nat.mod_eq_of_lt hl] at hmod; omega
        · rw [Nat.mod_eq_sub_mod hl, Nat.mod_eq_of_lt (by omega)] at hmod; omega
      omega
  · simp

theorem coeff_root_pow_lt (hn : 0 < n) {w : Nat} (hw : w < n) (i : Fin n) :
    Negacyclic.coeff (AdjoinRoot.root (negacyclicModulus n Int) ^ w) i =
      if i.val = w then 1 else 0 := by
  have := Negacyclic.coeff_root_pow (R := Int) hn ⟨w, hw⟩ i
  simp only at this
  rw [this]
  by_cases h : i.val = w
  · rw [if_pos (Fin.ext h.symm), if_pos h]
  · rw [if_neg (fun h' ↦ h (congrArg Fin.val h').symm), if_neg h]

theorem coeff_zero_root_pow_mul_const_lt (hn : 0 < n) (c : Int) {v : Nat} (hv : v < n) :
    Negacyclic.coeff (AdjoinRoot.root (negacyclicModulus n Int) ^ v * intPoly (fun _ ↦ c))
      ⟨0, hn⟩ = if v = 0 then c else -c := by
  rw [coeff_zero_mul hn]
  simp_rw [intPoly_coeff hn, coeff_root_pow_lt hn hv]
  rw [Finset.sum_eq_single ⟨(n - v) % n, Nat.mod_lt _ hn⟩]
  · have hidx : (n - (n - v) % n) % n = v := by
      by_cases hv0 : v = 0
      · subst hv0; simp [Nat.mod_self]
      · rw [Nat.mod_eq_of_lt (by omega : n - v < n), Nat.mod_eq_of_lt (by omega : n - (n - v) < n)]
        omega
    simp only [hidx, if_true, mul_one]
    by_cases hv0 : v = 0
    · have : (n - v) % n = 0 := by subst hv0; simp [Nat.mod_self]
      simp [hv0]
    · have : (n - v) % n ≠ 0 := by rw [Nat.mod_eq_of_lt (by omega)]; omega
      simp [this, hv0]
  · intro j _ hj
    have hne : (n - j.val) % n ≠ v := by
      intro h
      apply hj
      exact Fin.ext (by
        have hjl := j.isLt
        simp only
        by_cases hj0 : j.val = 0
        · rw [hj0, Nat.sub_zero, Nat.mod_self] at h
          subst h
          simp [hj0, Nat.mod_self]
        · rw [Nat.mod_eq_of_lt (by omega)] at h
          by_cases hv0 : v = 0
          · omega
          · rw [Nat.mod_eq_of_lt (by omega)]; omega)
    simp [hne]
  · simp

/-- The constant coefficient of `X^w` times the polynomial whose coefficients are all `c`. -/
theorem coeff_zero_root_pow_mul_const (hn : 0 < n) (c : Int) {w : Nat} (hw : w < 2 * n) :
    Negacyclic.coeff (AdjoinRoot.root (negacyclicModulus n Int) ^ w * intPoly (fun _ ↦ c))
      ⟨0, hn⟩ = if w = 0 ∨ n < w then c else -c := by
  by_cases hlt : w < n
  · rw [coeff_zero_root_pow_mul_const_lt hn c hlt]
    by_cases hw0 : w = 0
    · simp [hw0]
    · rw [if_neg hw0, if_neg (by omega)]
  · have hpow : AdjoinRoot.root (negacyclicModulus n Int) ^ w =
        -AdjoinRoot.root (negacyclicModulus n Int) ^ (w - n) := by
      rw [show w = n + (w - n) by omega, pow_add, Negacyclic.root_pow_n, Nat.add_sub_cancel_left,
        neg_one_mul]
    rw [hpow, neg_mul, Negacyclic.coeff_neg, coeff_zero_root_pow_mul_const_lt hn c (by omega)]
    by_cases hwn : w = n
    · rw [if_pos (by omega), if_neg (by omega)]
    · rw [if_neg (by omega), if_pos (Or.inr (by omega)), neg_neg]

/-- The root has order `2 n`. -/
theorem root_pow_mod_two_n {R : Type} [CommRing R] (k : Nat) :
    AdjoinRoot.root (negacyclicModulus n R) ^ k =
      AdjoinRoot.root (negacyclicModulus n R) ^ (k % (2 * n)) := by
  have hcycle : AdjoinRoot.root (negacyclicModulus n R) ^ (2 * n) = 1 := by
    rw [two_mul, pow_add, Negacyclic.root_pow_n]
    simp
  conv_lhs => rw [← Nat.div_add_mod k (2 * n), pow_add, pow_mul, hcycle, one_pow, one_mul]

/-- The constant coefficient of a constant table rotated by any power of the root. -/
theorem coeff_zero_root_pow_mul_const_mod (hn : 0 < n) (c : Int) (k : Nat) :
    Negacyclic.coeff (AdjoinRoot.root (negacyclicModulus n Int) ^ k * intPoly (fun _ ↦ c))
      ⟨0, hn⟩ = if k % (2 * n) = 0 ∨ n < k % (2 * n) then c else -c := by
  rw [root_pow_mod_two_n, coeff_zero_root_pow_mul_const hn c (Nat.mod_lt _ (by omega))]

end Negacyclic

/-- Base-`b` digits rebuild the low part of a natural number. -/
theorem digit_sum (b x : Nat) : ∀ k : Nat,
    ∑ d ∈ Finset.range k, (x / b ^ d % b) * b ^ d = x % b ^ k
  | 0 => by simp [Nat.mod_one]
  | k + 1 => by
    rw [Finset.sum_range_succ, digit_sum b x k, Nat.mod_pow_succ]
    ring

end MxxRuntime

