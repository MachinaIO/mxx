import TfheStages

/-!
# The closed form of public-key encryption

`encOf` is the encryption the `Encrypt` graph determines: the ring-LWE ciphertext
`(u a + e₁, u b + e₂ + (2 m - 1) ⌊Q/8⌋)`, the sample extraction of its constant coefficient
rescaled to `q` (`extractAt`), the base-4 digits of the leading 16 bits of the extracted mask
(`digitAt`) and the key switch `(-Aᵀ d, b' - Bᵀ d)`. The extraction and key-switching loops are
shared with the bootstrapped NAND gate.
-/

set_option maxRecDepth 16384

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime MxxSecurity GeneratedScheme

/-- Rescale `x mod Q` to `q`, rounding to nearest. -/
def rescale (x : Int) : Int :=
  (x % 5234636801 * 4294967296 + 2617318400) / 5234636801 % 4294967296

/-- The coefficient sample extraction reads for mask entry `i`: `-i mod N`. -/
def extractIndex (i : Fin 1024) : Fin 1024 := ⟨(1024 - i.val) % 1024, Nat.mod_lt _ (by norm_num)⟩

/-- Mask entry `i` of the extracted LWE ciphertext of a ring ciphertext with mask coefficients `c`,
rescaled to `q`. -/
def extractAt (c : Fin 1024 → Int) (i : Fin 1024) : Int :=
  rescale (c (extractIndex i) * ((if (i.val : Int) = 0 then 1 else 0) * 2 - 1))

/-- Key-switching digit `i`: digit `i % 8` in base 4 of the leading 16 bits of mask entry `i / 8`. -/
def digitAt (a : Fin 1024 → Int) (i : Fin 8192) : Int :=
  (a ⟨i.val / 8, by omega⟩ + 32768) / 65536 % 65536 / 4 ^ ((i.val : Int) % 8).toNat % 4

/-- The key-switched mask `-Aᵀ d mod q`. -/
def ksMask (A : Fin 5898240 → Int) (d : Fin 8192 → Int) (j : Fin 720) : Int :=
  (0 - intMatrixVectorProduct true A d j) % 4294967296

/-- The key-switched body `b - Bᵀ d mod q`. -/
def ksBody (B : Fin 8192 → Int) (b : Int) (d : Fin 8192 → Int) : Int :=
  (b - intMatrixVectorProduct (outer := 1) true B d 0) % 4294967296

theorem familyGet_eq' {α : Type} {count : Nat} {family : Fin count → α} {index : Int}
    {i : Fin count} (hi : (i.val : Int) = index) {output : α}
    (h : familyGetDynamic family index output) : output = family i := by
  subst hi
  exact familyGet_eq h

theorem extractIndex_val (i : Fin 1024) :
    ((extractIndex i).val : Int) = (1024 - (i.val : Int)) % 1024 := by
  simp only [extractIndex]
  have := i.isLt
  omega

/-- `select` over the eight powers of four. -/
theorem select_powers {k : Int} {output : Int}
    (h : select k [(1 : Int) + 0, 4 + 0, 16 + 0, 64 + 0, 256 + 0, 1024 + 0, 4096 + 0, 16384 + 0]
      output) :
    output = (4 : Int) ^ k.toNat := by
  obtain ⟨position, hposition, rfl⟩ := h
  subst hposition
  fin_cases position <;> rfl

theorem select_powers_holds {k : Int} (hk : 0 ≤ k ∧ k < 8) :
    select k [(1 : Int) + 0, 4 + 0, 16 + 0, 64 + 0, 256 + 0, 1024 + 0, 4096 + 0, 16384 + 0]
      ((4 : Int) ^ k.toNat) := by
  obtain ⟨p, rfl⟩ : ∃ p : Nat, k = p := ⟨k.toNat, by omega⟩
  have hp : p < 8 := by omega
  refine ⟨⟨p, by simpa using hp⟩, rfl, ?_⟩
  interval_cases p <;> rfl

/-! ## The loops of `Encrypt` -/

theorem encrypt_extract_iff (c : Fin 1024 → Int) (i : Fin 1024) (o : Int) :
    Encrypt.parallel_generatedRoot_8 Encrypt_params i (c, (4294967296, ())) o ↔
      o = extractAt c i := by
  constructor
  · rintro ⟨w6, _, _, _, hget, _, _, _, hout⟩
    rw [hout, familyGet_eq' (extractIndex_val i) hget]
    simp only [extractAt, rescale, decide_eq_true_eq]
    rfl
  · intro ho
    subst ho
    have hi := i.isLt
    refine ⟨c (extractIndex i), ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
    · norm_num
    · show (0 : Int) ≤ (1024 - (i.val : Int)) % 1024
      omega
    · show (1024 - (i.val : Int)) % 1024 < 1024
      omega
    · exact ⟨extractIndex i, extractIndex_val i, rfl⟩
    · norm_num
    · norm_num
    · norm_num
    · simp only [extractAt, rescale, decide_eq_true_eq]
      rfl

theorem digitIndex_val (i : Fin 8192) :
    ((⟨i.val / 8, by omega⟩ : Fin 1024).val : Int) = (i.val : Int) / 8 := by
  simp only
  omega

theorem encrypt_digit_iff (a : Fin 1024 → Int) (i : Fin 8192) (o : Int) :
    Encrypt.parallel_generatedRoot_17 Encrypt_params i
      (a, (1, (4, (16, (64, (256, (1024, (4096, (16384, ()))))))))) o ↔ o = digitAt a i := by
  constructor
  · rintro ⟨w4, w37, _, _, _, hget, _, _, _, _, _, _, hsel, _, _, hout⟩
    rw [hout, familyGet_eq' (digitIndex_val i) hget, select_powers hsel]
    rfl
  · intro ho
    subst ho
    have hi := i.isLt
    refine ⟨a ⟨i.val / 8, by omega⟩, (4 : Int) ^ (((i.val : Int) % 8).toNat), ?_⟩
    refine ⟨by norm_num, ?_, ?_, ⟨_, digitIndex_val i, rfl⟩, by norm_num, by norm_num, by norm_num,
      ?_, ?_, rfl, ?_, by positivity, by norm_num, rfl⟩
    · show (0 : Int) ≤ (i.val : Int) / 8; omega
    · show (i.val : Int) / 8 < 1024; omega
    · show (0 : Int) ≤ (i.val : Int) % 8; omega
    · show (i.val : Int) % 8 < 8; omega
    · have h := select_powers_holds (k := (i.val : Int) % 8) ⟨by omega, by omega⟩
      dsimp only
      exact h

theorem encrypt_neg_iff (x : Int) (i : Fin 720) (o : Int) :
    Encrypt.parallel_generatedRoot_20 Encrypt_params i (x, (4294967296, ())) o ↔
      o = (0 - x) % 4294967296 := by
  constructor
  · rintro ⟨_, hout⟩
    exact hout
  · intro ho
    exact ⟨by norm_num, ho⟩

/-! ## Sampled matrices satisfy their sampler relations -/

/-- The integer matrix of a sampled matrix. -/
noncomputable def smpInt {n rows columns : Nat} (law : SamplerLaw) (s : Nat) (ω : Key → ℤ) :
    ErrorMatrix n rows columns :=
  fun row column ↦ intPoly fun coefficient ↦
    law.clamp (ω (s, (finProdFinEquiv (finProdFinEquiv (row, column), coefficient)).val))

theorem smp_eq_reduce {q n rows columns : Nat} (law : SamplerLaw) (s : Nat) (ω : Key → ℤ) :
    (smp law s ω : ExactMatrix q n rows columns) = reduceMatrix q n rows columns (smpInt law s ω) := by
  funext row column
  exact polynomialOfCoefficients_eq_reduce _

theorem smpInt_coeff {n rows columns : Nat} (hn : 0 < n) (law : SamplerLaw) (s : Nat)
    (ω : Key → ℤ) (row : Fin rows) (column : Fin columns) (k : Fin n) :
    ((smpInt law s ω : ErrorMatrix n rows columns) row column).coeff k =
      law.clamp (ω (s, (finProdFinEquiv (finProdFinEquiv (row, column), k)).val)) :=
  intPoly_coeff hn _ k

theorem smp_interval_holds {q n rows columns : Nat} (hn : 0 < n) (s : Nat) (ω : Key → ℤ) :
    uniformIntervalSample 0 1 (smp bitLaw s ω : ExactMatrix q n rows columns) := by
  refine ⟨by norm_num, _, smp_eq_reduce _ _ _, fun row column k ↦ ?_⟩
  rw [smpInt_coeff hn]
  have := SamplerLaw.clamp_supports bitLaw
    (ω (s, (finProdFinEquiv (finProdFinEquiv (row, column), k)).val))
  simpa [SamplerLaw.Supports] using this

theorem smp_gaussian_holds {q n rows columns : Nat} (hn : 0 < n) {sigma : Rat} {cutoff : Int}
    (hsigma : 0 < sigma) (hcutoff : 0 ≤ cutoff) (s : Nat) (ω : Key → ℤ) :
    gaussianSample sigma cutoff
      (smp (.gaussian sigma cutoff) s ω : ExactMatrix q n rows columns) := by
  refine ⟨hsigma.le, hcutoff, ⟨_, smp_eq_reduce _ _ _, fun row column k ↦ ?_⟩,
    fun h ↦ absurd h hsigma.ne'⟩
  rw [smpInt_coeff hn]
  have := SamplerLaw.clamp_supports (.gaussian sigma cutoff)
    (ω (s, (finProdFinEquiv (finProdFinEquiv (row, column), k)).val))
  simp only [SamplerLaw.Supports, ne_eq, hsigma.ne', not_false_eq_true, hcutoff, and_self,
    ite_true] at this
  rw [abs_le] at this
  omega

/-! ## The encryption -/

/-- The canonical residues of the coefficients of a ring element. -/
noncomputable def coeffVals (x : ExactMatrix Q N 1 1) (k : Fin 1024) : Int := ((x 0 0).coeff k).val

theorem polynomialValues_coeffVals (x : ExactMatrix Q N 1 1) :
    polynomialValues false x (coeffVals x) := by
  simp only [polynomialValues, Bool.false_eq_true, if_false]
  intro i
  rfl

theorem polynomialValues_eq {x : ExactMatrix Q N 1 1} {v : Fin 1024 → Int}
    (h : polynomialValues false x v) : v = coeffVals x := by
  simp only [polynomialValues, Bool.false_eq_true, if_false] at h
  exact funext h

/-- The mask `u a + e₁` of the ring ciphertext. -/
noncomputable def encC0 (a u e1 : ExactMatrix Q N 1 1) : ExactMatrix Q N 1 1 :=
  matrixAdd (matrixMulScalarLeft u a) e1

/-- The body `u b + e₂ + (2 m - 1) ⌊Q/8⌋` of the ring ciphertext. -/
noncomputable def encC1 (b u e2 : ExactMatrix Q N 1 1) (m : Int) : ExactMatrix Q N 1 1 :=
  matrixAdd (matrixAdd (matrixMulScalarLeft u b) e2) (liftInteger ((m * 2 - 1) * 654329600))

/-- The inputs of `Encrypt`: the key-switching masks, the public key, the message and the
key-switching bodies. -/
abbrev EncInputs := (Fin 5898240 → Int) × ExactMatrix Q N 1 1 × ExactMatrix Q N 1 1 × Int ×
  (Fin 8192 → Int) × Unit

/-- The key-switching digits of an encryption. -/
noncomputable def encDigits (inputs : EncInputs) (u e1 : ExactMatrix Q N 1 1) : Fin 8192 → Int :=
  digitAt (extractAt (coeffVals (encC0 inputs.2.1 u e1)))

/-- The LWE ciphertext `Encrypt` returns. -/
noncomputable def encOf (inputs : EncInputs) (u e1 e2 : ExactMatrix Q N 1 1) :
    (Fin 720 → Int) × Int × Unit :=
  (ksMask inputs.1 (encDigits inputs u e1),
    ksBody inputs.2.2.2.2.1 (rescale (coeffVals (encC1 inputs.2.2.1 u e2 inputs.2.2.2.1) 0))
      (encDigits inputs u e1), ())

theorem encOut_eq (base : Nat) (inputs : EncInputs) (ω : Key → ℤ) :
    encOut base inputs ω =
      encOf inputs (smp bitLaw base ω) (smp ringLaw (base + 1) ω) (smp ringLaw (base + 2) ω) := by
  set u : ExactMatrix Q N 1 1 := smp bitLaw base ω
  set e1 : ExactMatrix Q N 1 1 := smp ringLaw (base + 1) ω
  set e2 : ExactMatrix Q N 1 1 := smp ringLaw (base + 2) ω
  set c0 := encC0 inputs.2.1 u e1
  set c1 := encC1 inputs.2.2.1 u e2 inputs.2.2.2.1
  set d := encDigits inputs u e1
  unfold encOut
  refine determined_eq ⟨⟨u, e1, coeffVals c0, extractAt (coeffVals c0), d, ksMask inputs.1 d, e2,
    coeffVals c1, coeffVals c1 0, intMatrixVectorProduct (outer := 1) true inputs.2.2.2.2.1 d 0⟩,
    rfl, rfl, rfl, ?_⟩ ?_
  · refine ⟨smp_interval_holds (by decide) _ _, smp_gaussian_holds (by decide) (by norm_num) (by norm_num) _ _,
      polynomialValues_coeffVals _, rfl, fun i ↦ (encrypt_extract_iff _ i _).mpr rfl, rfl,
      fun i ↦ (encrypt_digit_iff _ i _).mpr rfl, rfl, fun i ↦ (encrypt_neg_iff _ i _).mpr rfl,
      smp_gaussian_holds (by decide) (by norm_num) (by norm_num) _ _, polynomialValues_coeffVals _,
      familyGet_holds _ 0, by norm_num, by norm_num, by norm_num, familyGet_holds _ 0, by norm_num,
      rfl⟩
  · rintro other ⟨w, h1, h4, h23, _, _, h6, _, h8, _, h17, _, h20, _, h34, h35, _, _, _, h47, _,
      hout⟩
    obtain ⟨w1, w4, w6, w8, w17, w20, w23, w34, w35, w47⟩ := w
    simp only at h1 h4 h23
    subst h1 h4 h23
    have e6 : w6 = coeffVals c0 := polynomialValues_eq h6
    subst e6
    have e8 : w8 = extractAt (coeffVals c0) :=
      funext fun i ↦ (encrypt_extract_iff _ i _).mp (h8 i)
    subst e8
    have e17 : w17 = d := funext fun i ↦ (encrypt_digit_iff _ i _).mp (h17 i)
    subst e17
    have e20 : w20 = ksMask inputs.1 d := funext fun i ↦ (encrypt_neg_iff _ i _).mp (h20 i)
    subst e20
    have e34 : w34 = coeffVals c1 := polynomialValues_eq h34
    subst e34
    have e35 : w35 = coeffVals c1 0 := familyGet_eq (i := (0 : Fin 1024)) h35
    have e47 : w47 = intMatrixVectorProduct (outer := 1) true inputs.2.2.2.2.1 d 0 :=
      familyGet_eq (i := (0 : Fin 1)) h47
    subst e35 e47
    rw [hout]
    rfl

end MxxFheTfhe
