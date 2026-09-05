import ExponentRefreshBounds
import RuntimeModulusSwitch

namespace MxxExponentLut
open Mxx.Primitives
open scoped BigOperators

private theorem poly_ext {n : Nat} {R : Type} [CommRing R] [Nontrivial R]
    (hn : 0 < n) {a b : Negacyclic n R} (h : ∀ i, a.coeff i = b.coeff i) : a = b := by
  rw [Negacyclic.expansion hn a, Negacyclic.expansion hn b]
  simp_rw [h]

noncomputable def integerCoefficients {n : Nat} (values : Fin n → Int) : ErrorPoly n :=
  ∑ i, scaledBasis (values i) i

private theorem integerCoefficients_coeff {n : Nat} (hn : 0 < n)
    (values : Fin n → Int) (i : Fin n) :
    (integerCoefficients values).coeff i = values i := by
  rw [integerCoefficients, Negacyclic.coeff_sum]
  simp [CrtDecomposition.coeff_scaledBasis hn]

noncomputable def canonicalIntegerMatrix {q n rows columns : Nat}
    (input : ExactMatrix q n rows columns) : ErrorMatrix n rows columns :=
  fun row column => integerCoefficients (fun i => Int.ofNat ((input row column).coeff i).val)

private theorem canonicalIntegerMatrix_reduce {q n rows columns : Nat}
    (hq : 1 < q) (hn : 0 < n) (input : ExactMatrix q n rows columns) :
    reduceMatrix q n rows columns (canonicalIntegerMatrix input) = input := by
  letI : Fact (1 < q) := ⟨hq⟩
  ext row column
  apply poly_ext hn
  intro i
  simp only [reduceMatrix, canonicalIntegerMatrix]
  rw [reducePoly_coeff hq hn, integerCoefficients_coeff hn]
  simp

private theorem scalar_coeff {n : Nat} (scalar : Int) (poly : ErrorPoly n) (i : Fin n) :
    ((scalar : ErrorPoly n) * poly).coeff i = scalar * poly.coeff i := by
  exact Negacyclic.coeff_smul scalar poly i

private theorem nat_scalar_coeff {n : Nat} (scalar : Nat) (poly : ErrorPoly n) (i : Fin n) :
    ((scalar : ErrorPoly n) * poly).coeff i = (scalar : Int) * poly.coeff i := by
  simpa only [Int.cast_natCast] using scalar_coeff (scalar : Int) poly i

private theorem reduce_scalar {q n rows columns : Nat} (scalar : Int)
    (error : ErrorMatrix n rows columns) :
    reduceMatrix q n rows columns ((scalar : ErrorPoly n) • error) =
      (scalar : ExactPoly q n) • reduceMatrix q n rows columns error := by
  ext row column
  change reducePoly q n ((scalar : ErrorPoly n) * error row column) =
    (scalar : ExactPoly q n) * reducePoly q n (error row column)
  simp [reducePoly]

/-- Canonical coefficient congruence produces the actual source-modulus wrap
witness; divisibility is derived from reduction, never postulated. -/
theorem reduction_zero_integer_multiple {q n rows columns : Nat}
    (hq : 1 < q) (hn : 0 < n) (error : ErrorMatrix n rows columns)
    (hzero : reduceMatrix q n rows columns error = 0) :
    ∃ wrap : ErrorMatrix n rows columns, error = (q : ErrorPoly n) • wrap := by
  have hdvd (row : Fin rows) (column : Fin columns) (i : Fin n) :
      (q : Int) ∣ (error row column).coeff i := by
    apply (ZMod.intCast_zmod_eq_zero_iff_dvd _ q).mp
    have h := congrArg (fun e : ExactMatrix q n rows columns => (e row column).coeff i) hzero
    simpa only [reduceMatrix, reducePoly_coeff hq hn, Matrix.zero_apply, Negacyclic.coeff_zero] using h
  refine ⟨fun row column => integerCoefficients (fun i => (error row column).coeff i / (q : Int)), ?_⟩
  ext row column
  apply poly_ext hn
  intro i
  change (error row column).coeff i =
    (((q : Int) : ErrorPoly n) * integerCoefficients _).coeff i
  rw [scalar_coeff, integerCoefficients_coeff hn, Int.mul_ediv_cancel' (hdvd row column i)]

/-- Actual switch execution has a rounded integer matrix lift whose scaled
rounding residual is bounded coefficientwise by the odd lift's half-width. -/
theorem switch_integer_lift {q p n rows columns factor : Nat}
    (hn : 0 < n) (hf : 0 < factor) (hq : q = factor * p)
    (input : ExactMatrix q n rows columns) (output : ExactMatrix p n rows columns)
    (hrun : MxxRuntime.modulusSwitchRuns input output) :
    ∃ rounded : ErrorMatrix n rows columns,
      reduceMatrix p n rows columns rounded = output ∧
      CoeffBound ((factor : ErrorPoly n) • rounded - canonicalIntegerMatrix input) (factor / 2) := by
  classical
  have hw := MxxRuntime.modulusSwitchRuns_rounding_error hn hf hq input output hrun
  let coefficient := fun row column i => Classical.choose (hw row column i)
  have hs := fun row column i => Classical.choose_spec (hw row column i)
  refine ⟨fun row column => integerCoefficients (coefficient row column), ?_, ?_⟩
  · letI : Fact (1 < p) := ⟨hrun.1⟩
    ext row column
    apply poly_ext hn
    intro i
    simp only [reduceMatrix]
    rw [reducePoly_coeff hrun.1 hn, integerCoefficients_coeff hn]
    exact (hs row column i).1.symm
  · intro row column i
    change (((factor : ErrorPoly n) * integerCoefficients (coefficient row column) -
      integerCoefficients (fun k => Int.ofNat ((input row column).coeff k).val)).coeff i).natAbs ≤ _
    rw [Negacyclic.coeff_sub, nat_scalar_coeff, integerCoefficients_coeff hn,
      integerCoefficients_coeff hn]
    have hbounds := (hs row column i).2
    change -((factor : Int) / 2) ≤ (factor : Int) * coefficient row column i - _ ∧
      (factor : Int) * coefficient row column i - _ ≤ (factor : Int) / 2 at hbounds
    omega

private theorem reduce_sub {q n rows columns : Nat} (a b : ErrorMatrix n rows columns) :
    reduceMatrix q n rows columns (a - b) =
      reduceMatrix q n rows columns a - reduceMatrix q n rows columns b := by
  ext row column
  exact map_sub (reducePoly q n) _ _

private theorem reduce_nat_scalar {q n rows columns : Nat} (scalar : Nat)
    (error : ErrorMatrix n rows columns) :
    reduceMatrix q n rows columns ((scalar : ErrorPoly n) • error) =
      (scalar : ExactPoly q n) • reduceMatrix q n rows columns error := by
  simpa only [Int.cast_natCast] using reduce_scalar (q := q) (scalar : Int) error

private theorem modulus_cast_zero (p n : Nat) : (p : ExactPoly p n) = 0 := by
  have h := congrArg (algebraMap (ZMod p) (ExactPoly p n)) (ZMod.natCast_self p)
  simpa only [map_natCast, map_zero] using h

/-- The augmented secret contributes N coefficients from its small first
entry and exactly one unit entry, rather than a second convolution factor. -/
theorem augmented_secret_action_bound {n columns bound : Nat} (hn : 0 < n)
    (secret : ErrorMatrix n 1 2) (error : ErrorMatrix n 2 columns)
    (hsmall : ∀ i, ((secret 0 0).coeff i).natAbs ≤ 1)
    (hunit : secret 0 1 = 1) (herror : CoeffBound error bound) :
    CoeffBound (secret * error) ((n + 1) * bound) := by
  intro row column coefficient
  have hr : row = 0 := Subsingleton.elim _ _
  subst row
  have hs : polyNorm (secret 0 0) ≤ 1 := Finset.sup_le (fun i _ => hsmall i)
  have he (r : Fin 2) : polyNorm (error r column) ≤ bound :=
    Finset.sup_le (fun i _ => herror r column i)
  apply (coeff_natAbs_le_polyNorm _ _).trans
  change polyNorm (∑ r : Fin 2, secret 0 r * error r column) ≤ _
  rw [Fin.sum_univ_two, hunit, one_mul]
  calc
    _ ≤ polyNorm (secret 0 0 * error 0 column) + polyNorm (error 1 column) := polyNorm_add_le _ _
    _ ≤ n * 1 * bound + bound := Nat.add_le_add
      ((polyNorm_mul_le_tight hn _ _).trans
        (Nat.mul_le_mul (Nat.mul_le_mul_left n hs) (he 0))) (he 1)
    _ = _ := by ring

/-- Actual paired modulus switching, including source-modulus wraps. The
integer pair witness is constructed from canonical and rounded runtime lifts;
its scaled error estimate is a conclusion of the source encoding equation. -/
theorem paired_switch_error {q p n columns factor preBound : Nat}
    (hn : 0 < n) (hf : 0 < factor) (hq : q = factor * p)
    (secret : ErrorMatrix n 1 2) (ideal preError : ErrorMatrix n 1 columns)
    (ciphertext : ExactMatrix q n 1 columns) (publicKey : ExactMatrix q n 2 columns)
    (switchedCiphertext : ExactMatrix p n 1 columns) (switchedKey : ExactMatrix p n 2 columns)
    (hsmall : ∀ i, ((secret 0 0).coeff i).natAbs ≤ 1) (hunit : secret 0 1 = 1)
    (hpre : CoeffBound preError preBound)
    (hsource : ciphertext = reduceMatrix q n 1 2 secret * publicKey +
      reduceMatrix q n 1 columns ((factor : ErrorPoly n) • ideal + preError))
    (hc : MxxRuntime.modulusSwitchRuns ciphertext switchedCiphertext)
    (ha : MxxRuntime.modulusSwitchRuns publicKey switchedKey) :
    ∃ pairedError : ErrorMatrix n 1 columns,
      switchedCiphertext - reduceMatrix p n 1 2 secret * switchedKey -
        reduceMatrix p n 1 columns ideal = reduceMatrix p n 1 columns pairedError ∧
      ∀ row column coefficient, 2 * factor * ((pairedError row column).coeff coefficient).natAbs ≤
        2 * preBound + (factor - 1) * (n + 2) := by
  have hp : 1 < p := hc.1
  have hqgt : 1 < q := by nlinarith
  have hodd : factor % 2 = 1 := by
    have ho := hc.2.2.1
    rw [hq, Nat.mul_mod, hc.2.2.2.1, Nat.mul_one] at ho
    simpa only [Nat.mod_mod] using ho
  obtain ⟨roundedC, hroundedC, hboundC⟩ := switch_integer_lift hn hf hq ciphertext switchedCiphertext hc
  obtain ⟨roundedA, hroundedA, hboundA⟩ := switch_integer_lift hn hf hq publicKey switchedKey ha
  let canonicalC := canonicalIntegerMatrix ciphertext
  let canonicalA := canonicalIntegerMatrix publicKey
  have hcanonicalC := canonicalIntegerMatrix_reduce hqgt hn ciphertext
  have hcanonicalA := canonicalIntegerMatrix_reduce hqgt hn publicKey
  have hzero : reduceMatrix q n 1 columns
      (canonicalC - secret * canonicalA - (factor : ErrorPoly n) • ideal - preError) = 0 := by
    rw [reduce_sub, reduce_sub, reduce_sub, reduceMatrix_mul, reduce_nat_scalar,
      hcanonicalC, hcanonicalA, hsource, reduceMatrix_add, reduce_nat_scalar]
    abel
  obtain ⟨wrap, hwrap⟩ := reduction_zero_integer_multiple hqgt hn _ hzero
  let pairedError := roundedC - secret * roundedA - ideal - (p : ErrorPoly n) • wrap
  refine ⟨pairedError, ?_, ?_⟩
  · dsimp [pairedError]
    rw [reduce_sub, reduce_sub, reduce_sub, reduceMatrix_mul, reduce_nat_scalar,
      hroundedC, hroundedA, modulus_cast_zero, zero_smul, sub_zero]
  · have hscaled : (factor : ErrorPoly n) • pairedError = preError +
        ((factor : ErrorPoly n) • roundedC - canonicalC) -
        secret * ((factor : ErrorPoly n) • roundedA - canonicalA) := by
      dsimp [pairedError]
      rw [smul_sub, smul_sub, smul_sub, smul_smul, Matrix.mul_sub, Matrix.mul_smul]
      rw [show (q : ErrorPoly n) = (factor : ErrorPoly n) * (p : ErrorPoly n) by rw [hq, Nat.cast_mul]] at hwrap
      rw [← hwrap]
      abel
    have haction := augmented_secret_action_bound hn secret _ hsmall hunit hboundA
    have htotal : CoeffBound (preError +
        ((factor : ErrorPoly n) • roundedC - canonicalC) -
        secret * ((factor : ErrorPoly n) • roundedA - canonicalA))
        (preBound + factor / 2 + (n + 1) * (factor / 2)) := by
      intro row column coefficient
      change (((preError + ((factor : ErrorPoly n) • roundedC - canonicalC)) row column -
        (secret * ((factor : ErrorPoly n) • roundedA - canonicalA)) row column).coeff coefficient).natAbs ≤ _
      rw [Negacyclic.coeff_sub]
      exact (Int.natAbs_sub_le _ _).trans (Nat.add_le_add
        (coeffBound_add hpre hboundC row column coefficient) (haction row column coefficient))
    rw [← hscaled] at htotal
    intro row column coefficient
    have hb := htotal row column coefficient
    change (((factor : ErrorPoly n) * pairedError row column).coeff coefficient).natAbs ≤ _ at hb
    rw [nat_scalar_coeff, Int.natAbs_mul, Int.natAbs_natCast] at hb
    have hhalf : 2 * (factor / 2) = factor - 1 := by omega
    nlinarith

end MxxExponentLut
