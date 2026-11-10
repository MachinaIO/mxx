import TfheEncrypt

/-!
# The closed form of key generation

`keygenOf ω` is the output `Keygen` determines on the tape `ω`: the bootstrapping key (ring-GSW
encryptions of the LWE secret under the ring secret `z`), the key-switching key (LWE encryptions of
`z`'s gadget multiples under the LWE secret, with masks reduced from a ring of about `2^162`), the
public key `(a, a z + e)` and the LWE secret.
-/

set_option maxRecDepth 16384

namespace MxxFheTfhe

/-- Close a linear goal over `Int.ofNat` casts and `Fin` values. -/
macro "cast_omega" : tactic =>
  `(tactic| first | omega | (simp only [Int.ofNat_eq_coe, Fin.val_mk] at *; omega) | (simp; done))

open Mxx.Primitives MxxRuntime MxxSecurity GeneratedScheme

/-! ## Slices of one-row matrices -/

theorem slice_row_iff {q n cols w : Nat} (input : ExactMatrix q n 1 cols) (cs ce : Int) (off : Nat)
    (hcs : cs = off) (hce : ce = (w : Int) + cs) (hle : off + w ≤ cols)
    (out : ExactMatrix q n 1 w) :
    sliceMatrix input 0 1 cs ce out ↔ out = fun r c ↦ input r ⟨off + c.val, by omega⟩ := by
  subst hcs hce
  constructor
  · rintro ⟨-, -, -, -, -, -, -, -, h⟩
    funext r c
    rw [h r c (by have := r.isLt; omega) (by have := c.isLt; omega)]
    congr 1
    exact Subsingleton.elim _ _
  · rintro rfl
    refine ⟨le_refl _, by norm_num, by simp, by omega, by omega, ?_, by simp, by simp,
      fun r c _ _ ↦ ?_⟩
    · simp only [Int.ofNat_eq_coe]; omega
    · congr 1
      exact Subsingleton.elim _ _

/-! ## The secret and mask loops -/

theorem keygen_secret_iff (v : Fin 1024 → Int) (i : Fin 720) (o : Int) :
    Keygen.parallel_generatedRoot_3 Backend.backend Keygen_params i
      (v (Fin.castLE Keygen.index_le_720_1024 i)) o ↔
      o = v (Fin.castLE Keygen.index_le_720_1024 i) % 2 := by
  constructor
  · rintro ⟨_, h⟩; exact h
  · intro h; exact ⟨by norm_num, h⟩

/-- The canonical residues of the coefficients of mask column `c`. -/
noncomputable def maskVals (W : ExactMatrix Qmask N 1 5760) (c : Fin 5760) (k : Fin 1024) : Int :=
  ((W 0 c).coeff k).val

theorem keygen_maskcol_iff (W : ExactMatrix Qmask N 1 5760) (i : Fin 5760) (o : Fin 1024 → Int) :
    Keygen.parallel_generatedRoot_8 Backend.backend Keygen_params i W o ↔ o = maskVals W i := by
  have hle : i.val + 1 ≤ 5760 := i.isLt
  constructor
  · rintro ⟨w1, w2, -, -, -, -, -, -, hs, hv, hout⟩
    have hs' := (slice_row_iff W _ _ i.val rfl (by simp) hle w1).mp hs
    subst hs' hout
    funext k
    rw [hv k]
    rfl
  · rintro rfl
    refine ⟨fun r c ↦ W r ⟨i.val + c.val, by omega⟩, maskVals W i, by norm_num, by norm_num,
      by norm_num, ?_, ?_, ?_, (slice_row_iff W _ _ i.val rfl (by simp) hle _).mpr rfl,
      ?_, rfl⟩
    rotate_left
    all_goals try cast_omega
    simp only [polynomialValues, Bool.false_eq_true, if_false]
    intro k
    show maskVals W i k = _
    simp only [maskVals]
    rfl

/-- Key-switching mask entry `f`: coefficient `f % N` of mask column `f / N`, reduced mod `q`. -/
noncomputable def kskMaskOf (W : ExactMatrix Qmask N 1 5760) (f : Fin 5898240) : Int :=
  maskVals W ⟨f.val / 1024, by omega⟩ ⟨f.val % 1024, by omega⟩ % 4294967296

theorem keygen_mask_iff (W : ExactMatrix Qmask N 1 5760) (f : Fin 5898240) (o : Int) :
    Keygen.parallel_generatedRoot_10 Backend.backend Keygen_params f
      (maskVals W, (4294967296, ())) o ↔ o = kskMaskOf W f := by
  have hf := f.isLt
  constructor
  · rintro ⟨w4, w7, -, -, -, h4, -, -, -, h7, -, hout⟩
    have e4 : w4 = maskVals W ⟨f.val / 1024, by omega⟩ :=
      familyGet_eq' (by cast_omega) h4
    subst e4
    have e7 : w7 = maskVals W ⟨f.val / 1024, by omega⟩ ⟨f.val % 1024, by omega⟩ :=
      familyGet_eq' (by cast_omega) h7
    rw [hout, e7]
    rfl
  · rintro rfl
    refine ⟨maskVals W ⟨f.val / 1024, by omega⟩,
      maskVals W ⟨f.val / 1024, by omega⟩ ⟨f.val % 1024, by omega⟩, by norm_num, ?_, ?_,
      ⟨_, by cast_omega, rfl⟩, by norm_num, ?_, ?_, ⟨_, by cast_omega, rfl⟩, by norm_num, rfl⟩
    all_goals cast_omega

/-! ## The key-switching bodies -/

theorem select8_eq {f : Fin 8 → Int} {k : Int} {o : Int}
    (h : select k [f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7] o) :
    ∃ p : Fin 8, (p.val : Int) = k ∧ o = f p := by
  obtain ⟨position, hposition, rfl⟩ := h
  refine ⟨⟨position.val, by simpa using position.isLt⟩, hposition, ?_⟩
  fin_cases position <;> rfl

theorem select8_holds (f : Fin 8 → Int) (p : Fin 8) :
    select (p.val : Int) [f 0, f 1, f 2, f 3, f 4, f 5, f 6, f 7] (f p) := by
  refine ⟨⟨p.val, by simp⟩, rfl, ?_⟩
  fin_cases p <;> rfl

/-- The gadget multiples `2^16 4^k` of the key-switching key. -/
def kskPow (k : Fin 8) : Int := 65536 * 4 ^ k.val

/-- The centered value of a residue modulo `Q`. -/
def centerQ (x : Int) : Int := if x * 2 ≤ 5234636801 then x else x - 5234636801

/-- Key-switching body `i` from `⟨A_i, s⟩`: `⟨A_i, s⟩ + z_{i/8} 2^16 4^{i%8} + e_i mod q`, with the
error the centered coefficient `i % N` of error polynomial `i / N`. -/
def kskBodyAt (m : Int) (z : Fin 1024 → Int) (errs : Fin 8 → Fin 1024 → Int) (i : Fin 8192) : Int :=
  (m + z ⟨i.val / 8, by omega⟩ * kskPow ⟨i.val % 8, by omega⟩ +
    centerQ (errs ⟨i.val / 1024, by omega⟩ ⟨i.val % 1024, by omega⟩)) % 4294967296

theorem keygen_kskbody_iff (m : Int) (z : Fin 1024 → Int) (errs : Fin 8 → Fin 1024 → Int)
    (i : Fin 8192) (o : Int) :
    Keygen.parallel_generatedRoot_37 Backend.backend Keygen_params i
      (m, (z, (65536, (262144, (1048576, (4194304, (16777216, (67108864, (268435456,
        (1073741824, (errs 0, (errs 1, (errs 2, (errs 3, (errs 4, (errs 5, (errs 6, (errs 7,
          ())))))))))))))))))) o ↔ o = kskBodyAt m z errs i := by
  have hi := i.isLt
  constructor
  · rintro ⟨w5, w32, w40, w42, w44, w46, w48, w50, w52, w54, w55, w63, -, -, -, h5, -, -, -, -,
      h32, -, -, -, -, h40, -, -, h42, -, -, h44, -, -, h46, -, -, h48, -, -, h50, -, -, h52,
      -, -, h54, -, -, -, h55, -, -, -, h63, -, hout⟩
    have j : Fin 1024 := ⟨i.val % 1024, by omega⟩
    have hj : ((⟨i.val % 1024, by omega⟩ : Fin 1024).val : Int) = Int.ofNat i.val % 1024 := by
      cast_omega
    have e5 : w5 = z ⟨i.val / 8, by omega⟩ := familyGet_eq' (by cast_omega) h5
    have e32 : w32 = kskPow ⟨i.val % 8, by omega⟩ + 0 := by
      obtain ⟨p, hp, rfl⟩ := select8_eq (f := fun k ↦ kskPow k + 0) h32
      congr 2
      exact Fin.ext (by cast_omega)
    have e40 : w40 = errs 0 ⟨i.val % 1024, by omega⟩ := familyGet_eq' hj h40
    have e42 : w42 = errs 1 ⟨i.val % 1024, by omega⟩ := familyGet_eq' hj h42
    have e44 : w44 = errs 2 ⟨i.val % 1024, by omega⟩ := familyGet_eq' hj h44
    have e46 : w46 = errs 3 ⟨i.val % 1024, by omega⟩ := familyGet_eq' hj h46
    have e48 : w48 = errs 4 ⟨i.val % 1024, by omega⟩ := familyGet_eq' hj h48
    have e50 : w50 = errs 5 ⟨i.val % 1024, by omega⟩ := familyGet_eq' hj h50
    have e52 : w52 = errs 6 ⟨i.val % 1024, by omega⟩ := familyGet_eq' hj h52
    have e54 : w54 = errs 7 ⟨i.val % 1024, by omega⟩ := familyGet_eq' hj h54
    subst e40 e42 e44 e46 e48 e50 e52 e54
    have e55 : w55 = errs ⟨i.val / 1024, by omega⟩ ⟨i.val % 1024, by omega⟩ := by
      obtain ⟨p, hp, rfl⟩ := select8_eq (f := fun t ↦ errs t ⟨i.val % 1024, by omega⟩) h55
      congr 1
      exact Fin.ext (by cast_omega)
    have e63 := select_two h63
    rw [hout, e63, e5, e32, e55]
    simp only [kskBodyAt, centerQ, add_zero]
  · intro ho
    subst ho
    have hj : ((⟨i.val % 1024, by omega⟩ : Fin 1024).val : Int) = Int.ofNat i.val % 1024 := by
      cast_omega
    let e := errs ⟨i.val / 1024, by omega⟩ ⟨i.val % 1024, by omega⟩
    refine ⟨z ⟨i.val / 8, by omega⟩, kskPow ⟨i.val % 8, by omega⟩ + 0,
      errs 0 ⟨i.val % 1024, by omega⟩, errs 1 ⟨i.val % 1024, by omega⟩,
      errs 2 ⟨i.val % 1024, by omega⟩, errs 3 ⟨i.val % 1024, by omega⟩,
      errs 4 ⟨i.val % 1024, by omega⟩, errs 5 ⟨i.val % 1024, by omega⟩,
      errs 6 ⟨i.val % 1024, by omega⟩, errs 7 ⟨i.val % 1024, by omega⟩, e, centerQ e, ?_⟩
    refine ⟨by norm_num, ?_, ?_, ⟨_, by cast_omega, rfl⟩, by norm_num, ?_, ?_, rfl, ?_, by norm_num,
      by norm_num, ?_, ?_, ⟨_, hj, rfl⟩, ?_, ?_, ⟨_, hj, rfl⟩, ?_, ?_, ⟨_, hj, rfl⟩, ?_, ?_,
      ⟨_, hj, rfl⟩, ?_, ?_, ⟨_, hj, rfl⟩, ?_, ?_, ⟨_, hj, rfl⟩, ?_, ?_, ⟨_, hj, rfl⟩, ?_, ?_,
      ⟨_, hj, rfl⟩, ?_, ?_, rfl, ?_, ?_, ?_, rfl, ?_, by norm_num, ?_⟩
    all_goals try cast_omega
    · have h := select8_holds (fun k ↦ kskPow k + 0) ⟨i.val % 8, by omega⟩
      have hk : ((⟨i.val % 8, by omega⟩ : Fin 8).val : Int) = Int.ofNat i.val % 8 := by cast_omega
      rw [hk] at h
      exact h
    · have h := select8_holds (fun t ↦ errs t ⟨i.val % 1024, by omega⟩) ⟨i.val / 1024, by omega⟩
      have hk : ((⟨i.val / 1024, by omega⟩ : Fin 8).val : Int) = Int.ofNat i.val / 1024 := by
        cast_omega
      rw [hk] at h
      exact h
    all_goals first
      | (unfold centerQ; exact select_two_holds)
      | rfl
      | (split <;> norm_num)

/-! ## The bootstrapping key -/

theorem backend_layout : Backend.backend.regularLayout Q N = some Backend.layout0 := by
  simp [Backend.backend]

theorem layout_exact : Backend.layout0.droppedModuli = 0 := rfl

theorem layout_digitCount : Backend.layout0.digitCount = 6 := by decide

/-- The gadget row of the bootstrapping keys. -/
noncomputable def gadgetRow : ExactMatrix Q N 1 6 :=
  castMatrixColumns (by rw [layout_digitCount]) (regularGadgetMatrix (rows := 1) Backend.layout0)

theorem gadgetMatrixRuns_eq {g : ExactMatrix Q N 1 6}
    (hg : gadgetMatrixRuns Backend.backend 64 6 g) : g = gadgetRow := by
  obtain ⟨layout, hl, _, _, _, hgeq⟩ := hg
  rw [backend_layout] at hl
  cases hl
  rw [hgeq]
  rfl

theorem gadgetMatrixRuns_holds : gadgetMatrixRuns Backend.backend 64 6 gadgetRow :=
  ⟨Backend.layout0, backend_layout, by decide, by rw [layout_digitCount]; rfl,
    by rw [layout_digitCount], rfl⟩

/-- Columns `left` then `right`. -/
def concatOf {q n rows lc rc : Nat} (left : ExactMatrix q n rows lc) (right : ExactMatrix q n rows rc) :
    ExactMatrix q n rows (lc + rc) :=
  fun row column ↦ if h : column.val < lc then left row ⟨column.val, h⟩
    else right row ⟨column.val - lc, by omega⟩

theorem concat_iff {q n rows lc rc : Nat} (left : ExactMatrix q n rows lc)
    (right : ExactMatrix q n rows rc) (out : ExactMatrix q n rows (lc + rc)) :
    concatColumns left right out ↔ out = concatOf left right := by
  constructor
  · intro h
    funext row column
    exact h row column
  · rintro rfl row column
    rfl

/-- Bootstrapping key `i`: the rows `(a + [s g | 0], z a + E + [0 | s g])` of the ring-GSW
encryption of `s` under `z`, with mask `a` and error `E` the columns `[12 i, 12 i + 12)` of the
sampled matrices. -/
noncomputable def gswOf (A E : ExactMatrix Q N 1 8640) (z : ExactMatrix Q N 1 1) (s : Int)
    (i : Fin 720) : ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 × Unit :=
  let Ai : ExactMatrix Q N 1 12 := fun r c ↦ A r ⟨12 * i.val + c.val, by omega⟩
  let Ei : ExactMatrix Q N 1 12 := fun r c ↦ E r ⟨12 * i.val + c.val, by omega⟩
  let sg : ExactMatrix Q N 1 6 := matrixMulScalarLeft (liftInteger s) gadgetRow
  (matrixAdd Ai (concatOf sg (0 : ExactMatrix Q N 1 6)),
    matrixAdd (matrixAdd (matrixMulScalarLeft z Ai) Ei) (concatOf (0 : ExactMatrix Q N 1 6) sg), ())

theorem keygen_gsw_iff (A E : ExactMatrix Q N 1 8640) (z : ExactMatrix Q N 1 1) (s : Int)
    (i : Fin 720) (o : ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 × Unit) :
    Keygen.parallel_generatedRoot_6 Backend.backend Keygen_params i (A, (s, (z, (E, ())))) o ↔
      o = gswOf A E z s i := by
  have hi := i.isLt
  have hle : 12 * i.val + 12 ≤ 8640 := by omega
  constructor
  · rintro ⟨w1, w4, w7, w12, w14, -, -, -, -, -, -, h1, h4, h7, -, -, -, -, -, -, h12, h14, hout⟩
    have e1 := (slice_row_iff A _ _ (12 * i.val) (by cast_omega) (by cast_omega) hle w1).mp h1
    have e12 := (slice_row_iff E _ _ (12 * i.val) (by cast_omega) (by cast_omega) hle w12).mp h12
    have e4 := gadgetMatrixRuns_eq h4
    subst e1 e12 e4
    have e7 := (concat_iff _ _ _).mp h7
    have e14 := (concat_iff _ _ _).mp h14
    subst e7 e14
    rw [hout]
    rfl
  · rintro rfl
    refine ⟨_, gadgetRow, _, _, _, by norm_num, by norm_num, by norm_num, ?_, ?_, ?_,
      (slice_row_iff A _ _ (12 * i.val) (by cast_omega) (by cast_omega) hle _).mpr rfl,
      gadgetMatrixRuns_holds, (concat_iff _ _ _).mpr rfl, by norm_num, by norm_num, by norm_num,
      ?_, ?_, ?_, (slice_row_iff E _ _ (12 * i.val) (by cast_omega) (by cast_omega) hle _).mpr rfl,
      (concat_iff _ _ _).mpr rfl, rfl⟩
    all_goals cast_omega

/-! ## Key generation -/

/-- The LWE secret: coefficient `j` of the sampled binary ring element, reduced modulo 2. -/
noncomputable def secretOf (ω : Key → ℤ) (j : Fin 720) : Int :=
  coeffVals (smp bitLaw 1 ω) (Fin.castLE Keygen.index_le_720_1024 j) % 2

/-- The coefficients of the key-switching error polynomials. -/
noncomputable def kskErrs (ω : Key → ℤ) (k : Fin 8) : Fin 1024 → Int :=
  coeffVals (smp lweLaw (5 + k.val) ω)

/-- The key-switching masks. -/
noncomputable def kskAOf (ω : Key → ℤ) : Fin 5898240 → Int :=
  kskMaskOf (smp (.residue Qmask) 4 ω)

/-- The key-switching bodies. -/
noncomputable def kskBOf (ω : Key → ℤ) (i : Fin 8192) : Int :=
  kskBodyAt (intMatrixVectorProduct false (kskAOf ω) (secretOf ω) i)
    (coeffVals (smp bitLaw 2 ω)) (kskErrs ω) i

/-- Bootstrapping key `i`. -/
noncomputable def bskOf (ω : Key → ℤ) (i : Fin 720) :
    ExactMatrix Q N 1 12 × ExactMatrix Q N 1 12 × Unit :=
  gswOf (smp (.residue Q) 0 ω) (smp ringLaw 3 ω) (smp bitLaw 2 ω) (secretOf ω i) i

/-- The public-key body `a z + e`. -/
noncomputable def pkBOf (ω : Key → ℤ) : ExactMatrix Q N 1 1 :=
  matrixAdd (matrixMulScalarLeft (smp (.residue Q) 13 ω) (smp bitLaw 2 ω)) (smp ringLaw 14 ω)

/-- The output of `Keygen` on the tape `ω`. -/
noncomputable def keygenOf (ω : Key → ℤ) : KeygenOutputs :=
  (fun i ↦ (bskOf ω i).1, fun i ↦ (bskOf ω i).2.1, kskAOf ω, kskBOf ω, smp (.residue Q) 13 ω,
    pkBOf ω, kskAOf ω, kskBOf ω, secretOf ω, ())

theorem keygenOut_eq (ω : Key → ℤ) : keygenOut ω = keygenOf ω := by
  unfold keygenOut
  refine determined_eq ⟨⟨smp (.residue Q) 0 ω, smp bitLaw 1 ω, coeffVals (smp bitLaw 1 ω),
    secretOf ω, smp bitLaw 2 ω, smp ringLaw 3 ω, fun i ↦ (bskOf ω i).1, fun i ↦ (bskOf ω i).2.1,
    smp (.residue Qmask) 4 ω, maskVals (smp (.residue Qmask) 4 ω), kskAOf ω,
    coeffVals (smp bitLaw 2 ω), smp lweLaw 5 ω, kskErrs ω 0, smp lweLaw 6 ω, kskErrs ω 1,
    smp lweLaw 7 ω, kskErrs ω 2, smp lweLaw 8 ω, kskErrs ω 3, smp lweLaw 9 ω, kskErrs ω 4,
    smp lweLaw 10 ω, kskErrs ω 5, smp lweLaw 11 ω, kskErrs ω 6, smp lweLaw 12 ω, kskErrs ω 7,
    kskBOf ω, smp (.residue Q) 13 ω, smp ringLaw 14 ω⟩, rfl, rfl, rfl, rfl, rfl, rfl, rfl, rfl,
    rfl, rfl, rfl, rfl, rfl, rfl, rfl, ?_⟩ ?_
  · have hg := fun s : Nat ↦
      (smp_gaussian_holds (q := Q) (n := N) (rows := 1) (columns := 1) (by decide)
        (sigma := 32768 / 1) (cutoff := 524288) (by norm_num) (by norm_num) s ω)
    refine ⟨rfl, smp_interval_holds (by decide) _ _, polynomialValues_coeffVals _, rfl,
      fun i ↦ (keygen_secret_iff _ i _).mpr rfl, smp_interval_holds (by decide) _ _,
      smp_gaussian_holds (by decide) (by norm_num) (by norm_num) _ _, rfl,
      fun i ↦ (keygen_gsw_iff _ _ _ _ i _).mpr rfl, rfl, rfl,
      fun i ↦ (keygen_maskcol_iff _ i _).mpr rfl, rfl, fun f ↦ (keygen_mask_iff _ f _).mpr rfl,
      polynomialValues_coeffVals _, hg 5, polynomialValues_coeffVals _, hg 6,
      polynomialValues_coeffVals _, hg 7, polynomialValues_coeffVals _, hg 8,
      polynomialValues_coeffVals _, hg 9, polynomialValues_coeffVals _, hg 10,
      polynomialValues_coeffVals _, hg 11, polynomialValues_coeffVals _, hg 12,
      polynomialValues_coeffVals _, rfl, fun i ↦ (keygen_kskbody_iff _ _ (kskErrs ω) i _).mpr rfl,
      rfl, smp_gaussian_holds (by decide) (by norm_num) (by norm_num) _ _, rfl⟩
  · rintro other ⟨w, e0, e1, e4, e5, e7, e21, e23, e25, e27, e29, e31, e33, e35, e38, e40,
      -, -, h2, -, h3, -, -, -, h6, -, -, h8, -, h10, h12, -, h22, -, h24, -, h26, -, h28, -, h30,
      -, h32, -, h34, -, h36, -, h37, -, -, hout⟩
    obtain ⟨w0, w1, w2, w3, w4, w5, w60, w61, w7, w8, w10, w12, w21, w22, w23, w24, w25, w26, w27,
      w28, w29, w30, w31, w32, w33, w34, w35, w36, w37, w38, w40⟩ := w
    simp only at e0 e1 e4 e5 e7 e21 e23 e25 e27 e29 e31 e33 e35 e38 e40
    subst e0 e1 e4 e5 e7 e21 e23 e25 e27 e29 e31 e33 e35 e38 e40
    have f2 : w2 = coeffVals (smp bitLaw 1 ω) := polynomialValues_eq h2
    subst f2
    have f3 : w3 = secretOf ω := funext fun i ↦ (keygen_secret_iff _ i _).mp (h3 i)
    subst f3
    have f60 : w60 = fun i ↦ (bskOf ω i).1 :=
      funext fun i ↦ congrArg Prod.fst ((keygen_gsw_iff _ _ _ _ i _).mp (h6 i))
    have f61 : w61 = fun i ↦ (bskOf ω i).2.1 :=
      funext fun i ↦ congrArg (fun x ↦ x.2.1) ((keygen_gsw_iff _ _ _ _ i _).mp (h6 i))
    subst f60 f61
    have f8 : w8 = maskVals (smp (.residue Qmask) 4 ω) :=
      funext fun i ↦ (keygen_maskcol_iff _ i _).mp (h8 i)
    subst f8
    have f10 : w10 = kskAOf ω := funext fun f ↦ (keygen_mask_iff _ f _).mp (h10 f)
    subst f10
    have f12 : w12 = coeffVals (smp bitLaw 2 ω) := polynomialValues_eq h12
    have f22 : w22 = kskErrs ω 0 := polynomialValues_eq h22
    have f24 : w24 = kskErrs ω 1 := polynomialValues_eq h24
    have f26 : w26 = kskErrs ω 2 := polynomialValues_eq h26
    have f28 : w28 = kskErrs ω 3 := polynomialValues_eq h28
    have f30 : w30 = kskErrs ω 4 := polynomialValues_eq h30
    have f32 : w32 = kskErrs ω 5 := polynomialValues_eq h32
    have f34 : w34 = kskErrs ω 6 := polynomialValues_eq h34
    have f36 : w36 = kskErrs ω 7 := polynomialValues_eq h36
    subst f12 f22 f24 f26 f28 f30 f32 f34 f36
    have f37 : w37 = kskBOf ω := funext fun i ↦ (keygen_kskbody_iff _ _ (kskErrs ω) i _).mp (h37 i)
    subst f37
    rw [hout]
    rfl

end MxxFheTfhe
