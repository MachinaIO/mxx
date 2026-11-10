import Reductions
import BgvForms

set_option backward.isDefEq.respectTransparency false

/-! Closed forms of the generated BGV encryptions, and the samplers they satisfy. -/

namespace BgvSecurity
open MxxSecurity Mxx.Primitives MxxRuntime

/-- The integer matrix whose coefficients are `values`, as `matrixOf` lays them out. -/
noncomputable def intMatrixOf {rows columns : Nat} (values : Fin (rows * columns * n) → Int) :
    ErrorMatrix n rows columns :=
  fun row column ↦ intPoly fun k ↦ values (finProdFinEquiv (finProdFinEquiv (row, column), k))

theorem matrixOf_eq_reduce {q rows columns : Nat} (values : Fin (rows * columns * n) → Int) :
    matrixOf (q := q) values = reduceMatrix q n rows columns (intMatrixOf values) := by
  funext row column
  exact polynomialOfCoefficients_eq_reduce _

theorem intMatrixOf_coeff {rows columns : Nat} (values : Fin (rows * columns * n) → Int)
    (row : Fin rows) (column : Fin columns) (k : Fin n) :
    (intMatrixOf values row column).coeff k =
      values (finProdFinEquiv (finProdFinEquiv (row, column), k)) :=
  intPoly_coeff (by decide) _ k

theorem interval_holds {q rows columns : Nat} (values : Fin (rows * columns * n) → Int) :
    uniformIntervalSample (-1) 1
      (matrixOf (q := q) fun i ↦ (SamplerLaw.interval (-1) 1).clamp (values i)) := by
  refine ⟨by norm_num, _, matrixOf_eq_reduce _, fun row column k ↦ ?_⟩
  rw [intMatrixOf_coeff]
  have := SamplerLaw.clamp_supports (.interval (-1) 1)
    (values (finProdFinEquiv (finProdFinEquiv (row, column), k)))
  simpa [SamplerLaw.Supports] using this

theorem gaussian_holds {q rows columns : Nat} {sigma : Rat} {cutoff : Int} (hsigma : 0 < sigma)
    (hcutoff : 0 ≤ cutoff) (values : Fin (rows * columns * n) → Int) :
    gaussianSample sigma cutoff
      (matrixOf (q := q) fun i ↦ (SamplerLaw.gaussian sigma cutoff).clamp (values i)) := by
  refine ⟨hsigma.le, hcutoff, ⟨_, matrixOf_eq_reduce _, fun row column k ↦ ?_⟩,
    fun h ↦ absurd h hsigma.ne'⟩
  rw [intMatrixOf_coeff]
  have := SamplerLaw.clamp_supports (.gaussian sigma cutoff)
    (values (finProdFinEquiv (finProdFinEquiv (row, column), k)))
  simp only [SamplerLaw.Supports, ne_eq, hsigma.ne', not_false_eq_true, hcutoff, and_self,
    ite_true] at this
  rw [abs_le] at this
  omega

/-- The Gaussian law of every BGV error. -/
abbrev errLaw : SamplerLaw := .gaussian (3602879701896397/1125899906842624) 20
/-- The ternary law of every BGV secret. -/
abbrev ternLaw : SamplerLaw := .interval (-1) 1

/-- A clamped `1 × 1` sample. -/
noncomputable def mat11 {q : Nat} (law : SamplerLaw) (values : Fin (1 * 1 * n) → Int) :
    ExactMatrix q n 1 1 :=
  matrixOf fun i ↦ law.clamp (values i)

/-- The BGV encryption `(pk₀ U + t e₁, pk₁ U + t e₂ + E)`. -/
noncomputable def encOf (pk : ExactMatrix Q n 2 1) (U e1 e2 E : ExactMatrix Q n 1 1) :
    ExactMatrix Q n 2 1 :=
  fun r _ ↦ if r.val = 0 then pk 0 0 * U 0 0 + tPoly * e1 0 0
    else pk 1 0 * U 0 0 + tPoly * e2 0 0 + E 0 0

theorem slice_row0 {q : Nat} (a : ExactMatrix q n 2 1) :
    sliceMatrix a 0 1 0 1 (fun _ _ ↦ a 0 0 : ExactMatrix q n 1 1) := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    fun row column _ _ ↦ ?_⟩
  fin_cases row; fin_cases column; rfl

theorem slice_row1 {q : Nat} (a : ExactMatrix q n 2 1) :
    sliceMatrix a 1 2 0 1 (fun _ _ ↦ a 1 0 : ExactMatrix q n 1 1) := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    fun row column _ _ ↦ ?_⟩
  fin_cases row; fin_cases column; rfl

theorem mulScalarLeft_apply {q : Nat} (a b : ExactMatrix q n 1 1) :
    matrixMulScalarLeft a b 0 0 = a 0 0 * b 0 0 := rfl

theorem scale_apply {q r c : Nat} (k : Int) (b : ExactMatrix q n r c) (i : Fin r) (j : Fin c) :
    matrixScale k b i j = (k : ExactPoly q n) * b i j := rfl

theorem add_apply' {q r c : Nat} (a b : ExactMatrix q n r c) (i : Fin r) (j : Fin c) :
    matrixAdd a b i j = a i j + b i j := rfl

theorem encOf_zero (pk : ExactMatrix Q n 2 1) (U e1 e2 E : ExactMatrix Q n 1 1) :
    encOf pk U e1 e2 E 0 0 = pk 0 0 * U 0 0 + tPoly * e1 0 0 := if_pos rfl

theorem encOf_one (pk : ExactMatrix Q n 2 1) (U e1 e2 E : ExactMatrix Q n 1 1) :
    encOf pk U e1 e2 E 1 0 = pk 1 0 * U 0 0 + tPoly * e2 0 0 + E 0 0 := if_neg (by decide)

theorem table13_eq : Encrypt.generatedRoot.table_13 = encodeTable := rfl

theorem encrypt_eq (pk : ExactMatrix Q n 2 1) (m : Fin n → Int) :
    Encrypt.generatedRoot.comp GeneratedScheme.Encrypt_params (pk, m, ()) =
      (do let vU ← sampleFamily ternLaw (1 * 1 * n)
          let v1 ← sampleFamily errLaw (1 * 1 * n)
          let v2 ← sampleFamily errLaw (1 * 1 * n)
          pure (encOf pk (mat11 ternLaw vU) (mat11 errLaw v1) (mat11 errLaw v2) (encodeSlots m))) := by
  unfold Encrypt.generatedRoot.comp
  simp only [sampleInterval, sampleGaussian, sampleMatrix_clamp, bind_assoc, pure_bind]
  refine bind_congr fun vU ↦ bind_congr fun v1 ↦ bind_congr fun v2 ↦ ?_
  congr 1
  refine determined_eq ⟨⟨fun _ _ ↦ pk 0 0, mat11 ternLaw vU, mat11 errLaw v1, fun _ _ ↦ pk 1 0,
    mat11 errLaw v2, tableValues, nativeOf m, plainOf m, encodeSlots m,
    encOf pk (mat11 ternLaw vU) (mat11 errLaw v1) (mat11 errLaw v2) (encodeSlots m)⟩,
    by simp only [mat11, ternLaw], by simp only [mat11, errLaw], by simp only [mat11, errLaw], ?_⟩ ?_
  · dsimp only [Encrypt.generatedRoot.body]
    rw [table13_eq]
    refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, slice_row0 pk,
      interval_holds vU, gaussian_holds (by norm_num) (by norm_num) v1, by decide, by decide,
      by decide, by decide, by decide, by decide, slice_row1 pk,
      gaussian_holds (by norm_num) (by norm_num) v2, tableValues_holds, rfl,
      fun i ↦ ⟨nativeOf m i, ?_, ?_, slot_read_holds m i, rfl⟩, plainOf_spec m, ⟨Fact.out, rfl⟩,
      ?_, rfl⟩
    · rw [tableValues_eq]; exact Int.natCast_nonneg _
    · rw [tableValues_eq]; exact encodeEntry_lt_int i
    · refine concatRows_of_rows ?_ ?_
      · rw [encOf_zero, add_apply', mulScalarLeft_apply, scale_apply]
      · rw [encOf_one, add_apply', add_apply', mulScalarLeft_apply, scale_apply]
  · rintro outputs ⟨w, hU, h1, h2, h⟩
    dsimp only [Encrypt.generatedRoot.body] at h
    rw [table13_eq] at h
    obtain ⟨-, -, -, -, -, -, hs0, -, -, -, -, -, -, -, -, hs1, -, hvals, -, hpar, hfrom, hreb,
      hcat, rfl⟩ := h
    have h0 := sliceMatrix_entry hs0 (by decide) (by decide)
    have h1' := sliceMatrix_entry hs1 (by decide) (by decide)
    have htab := table_values _ hvals
    have hnative : w.w_15_0 = nativeOf m := by
      funext i
      obtain ⟨value, -, -, hget, hout⟩ := hpar i
      rw [hout]
      exact slot_read htab i hget
    rw [hnative] at hfrom
    have hplain : w.w_16_0 = plainOf m := fromValues_unique hfrom (plainOf_spec m)
    have henc : w.w_17_0 = encodeSlots m := by rw [hreb.2, hplain]; rfl
    have h0'' : w.w_1_0 0 0 = pk 0 0 := h0
    have h1'' : w.w_7_0 0 0 = pk 1 0 := h1'
    obtain ⟨c0, c1⟩ := concatRows_one_one hcat 0
    refine ext21 ?_ ?_
    · rw [c0, encOf_zero, add_apply', mulScalarLeft_apply, scale_apply, h0'', hU, h1]
      simp only [mat11, ternLaw, errLaw, tPoly]
    · rw [c1, encOf_one, add_apply', add_apply', mulScalarLeft_apply, scale_apply, h1'', hU, h2,
        henc]
      simp only [mat11, ternLaw, errLaw, tPoly]
theorem sim_0_encrypt_table : Sim_0_encrypt.generatedRoot.table_13 = encodeTable := rfl

theorem sim0_encrypt_eq (pk : ExactMatrix Q n 2 1) (m : Fin n → Int) :
    Sim_0_encrypt.generatedRoot.comp GeneratedScheme.Sim_0_encrypt_params (pk, m, ()) =
      (do let vU ← sampleFamily ternLaw (1 * 1 * n)
          let v1 ← sampleFamily errLaw (1 * 1 * n)
          let v2 ← sampleFamily errLaw (1 * 1 * n)
          pure (encOf pk (mat11 ternLaw vU) (mat11 errLaw v1) (mat11 errLaw v2) (encodeSlots m))) := by
  unfold Sim_0_encrypt.generatedRoot.comp
  simp only [sampleInterval, sampleGaussian, sampleMatrix_clamp, bind_assoc, pure_bind]
  refine bind_congr fun vU ↦ bind_congr fun v1 ↦ bind_congr fun v2 ↦ ?_
  congr 1
  refine determined_eq ⟨⟨fun _ _ ↦ pk 0 0, mat11 ternLaw vU, mat11 errLaw v1, fun _ _ ↦ pk 1 0,
    mat11 errLaw v2, tableValues, nativeOf m, plainOf m, encodeSlots m,
    encOf pk (mat11 ternLaw vU) (mat11 errLaw v1) (mat11 errLaw v2) (encodeSlots m)⟩,
    by simp only [mat11, ternLaw], by simp only [mat11, errLaw], by simp only [mat11, errLaw], ?_⟩ ?_
  · dsimp only [Sim_0_encrypt.generatedRoot.body]
    rw [sim_0_encrypt_table]
    refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, slice_row0 pk,
      interval_holds vU, gaussian_holds (by norm_num) (by norm_num) v1, by decide, by decide,
      by decide, by decide, by decide, by decide, slice_row1 pk,
      gaussian_holds (by norm_num) (by norm_num) v2, tableValues_holds, rfl,
      fun i ↦ ⟨nativeOf m i, ?_, ?_, slot_read_holds m i, rfl⟩, plainOf_spec m, ⟨Fact.out, rfl⟩,
      ?_, rfl⟩
    · rw [tableValues_eq]; exact Int.natCast_nonneg _
    · rw [tableValues_eq]; exact encodeEntry_lt_int i
    · refine concatRows_of_rows ?_ ?_
      · rw [encOf_zero, add_apply', mulScalarLeft_apply, scale_apply]
      · rw [encOf_one, add_apply', add_apply', mulScalarLeft_apply, scale_apply]
  · rintro outputs ⟨w, hU, h1, h2, h⟩
    dsimp only [Sim_0_encrypt.generatedRoot.body] at h
    rw [sim_0_encrypt_table] at h
    obtain ⟨-, -, -, -, -, -, hs0, -, -, -, -, -, -, -, -, hs1, -, hvals, -, hpar, hfrom, hreb,
      hcat, rfl⟩ := h
    have h0 := sliceMatrix_entry hs0 (by decide) (by decide)
    have h1' := sliceMatrix_entry hs1 (by decide) (by decide)
    have htab := table_values _ hvals
    have hnative : w.w_15_0 = nativeOf m := by
      funext i
      obtain ⟨value, -, -, hget, hout⟩ := hpar i
      rw [hout]
      exact slot_read htab i hget
    rw [hnative] at hfrom
    have hplain : w.w_16_0 = plainOf m := fromValues_unique hfrom (plainOf_spec m)
    have henc : w.w_17_0 = encodeSlots m := by rw [hreb.2, hplain]; rfl
    have h0'' : w.w_1_0 0 0 = pk 0 0 := h0
    have h1'' : w.w_7_0 0 0 = pk 1 0 := h1'
    obtain ⟨c0, c1⟩ := concatRows_one_one hcat 0
    refine ext21 ?_ ?_
    · rw [c0, encOf_zero, add_apply', mulScalarLeft_apply, scale_apply, h0'', hU, h1]
      simp only [mat11, ternLaw, errLaw, tPoly]
    · rw [c1, encOf_one, add_apply', add_apply', mulScalarLeft_apply, scale_apply, h1'', hU, h2,
        henc]
      simp only [mat11, ternLaw, errLaw, tPoly]
theorem sim_1_encrypt_table : Sim_1_encrypt.generatedRoot.table_13 = encodeTable := rfl

theorem sim1_encrypt_eq (pk : ExactMatrix Q n 2 1) (m : Fin n → Int) :
    Sim_1_encrypt.generatedRoot.comp GeneratedScheme.Sim_1_encrypt_params (pk, m, ()) =
      (do let vU ← sampleFamily ternLaw (1 * 1 * n)
          let v1 ← sampleFamily errLaw (1 * 1 * n)
          let v2 ← sampleFamily errLaw (1 * 1 * n)
          pure (encOf pk (mat11 ternLaw vU) (mat11 errLaw v1) (mat11 errLaw v2) (encodeSlots m))) := by
  unfold Sim_1_encrypt.generatedRoot.comp
  simp only [sampleInterval, sampleGaussian, sampleMatrix_clamp, bind_assoc, pure_bind]
  refine bind_congr fun vU ↦ bind_congr fun v1 ↦ bind_congr fun v2 ↦ ?_
  congr 1
  refine determined_eq ⟨⟨fun _ _ ↦ pk 0 0, mat11 ternLaw vU, mat11 errLaw v1, fun _ _ ↦ pk 1 0,
    mat11 errLaw v2, tableValues, nativeOf m, plainOf m, encodeSlots m,
    encOf pk (mat11 ternLaw vU) (mat11 errLaw v1) (mat11 errLaw v2) (encodeSlots m)⟩,
    by simp only [mat11, ternLaw], by simp only [mat11, errLaw], by simp only [mat11, errLaw], ?_⟩ ?_
  · dsimp only [Sim_1_encrypt.generatedRoot.body]
    rw [sim_1_encrypt_table]
    refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, slice_row0 pk,
      interval_holds vU, gaussian_holds (by norm_num) (by norm_num) v1, by decide, by decide,
      by decide, by decide, by decide, by decide, slice_row1 pk,
      gaussian_holds (by norm_num) (by norm_num) v2, tableValues_holds, rfl,
      fun i ↦ ⟨nativeOf m i, ?_, ?_, slot_read_holds m i, rfl⟩, plainOf_spec m, ⟨Fact.out, rfl⟩,
      ?_, rfl⟩
    · rw [tableValues_eq]; exact Int.natCast_nonneg _
    · rw [tableValues_eq]; exact encodeEntry_lt_int i
    · refine concatRows_of_rows ?_ ?_
      · rw [encOf_zero, add_apply', mulScalarLeft_apply, scale_apply]
      · rw [encOf_one, add_apply', add_apply', mulScalarLeft_apply, scale_apply]
  · rintro outputs ⟨w, hU, h1, h2, h⟩
    dsimp only [Sim_1_encrypt.generatedRoot.body] at h
    rw [sim_1_encrypt_table] at h
    obtain ⟨-, -, -, -, -, -, hs0, -, -, -, -, -, -, -, -, hs1, -, hvals, -, hpar, hfrom, hreb,
      hcat, rfl⟩ := h
    have h0 := sliceMatrix_entry hs0 (by decide) (by decide)
    have h1' := sliceMatrix_entry hs1 (by decide) (by decide)
    have htab := table_values _ hvals
    have hnative : w.w_15_0 = nativeOf m := by
      funext i
      obtain ⟨value, -, -, hget, hout⟩ := hpar i
      rw [hout]
      exact slot_read htab i hget
    rw [hnative] at hfrom
    have hplain : w.w_16_0 = plainOf m := fromValues_unique hfrom (plainOf_spec m)
    have henc : w.w_17_0 = encodeSlots m := by rw [hreb.2, hplain]; rfl
    have h0'' : w.w_1_0 0 0 = pk 0 0 := h0
    have h1'' : w.w_7_0 0 0 = pk 1 0 := h1'
    obtain ⟨c0, c1⟩ := concatRows_one_one hcat 0
    refine ext21 ?_ ?_
    · rw [c0, encOf_zero, add_apply', mulScalarLeft_apply, scale_apply, h0'', hU, h1]
      simp only [mat11, ternLaw, errLaw, tPoly]
    · rw [c1, encOf_one, add_apply', add_apply', mulScalarLeft_apply, scale_apply, h1'', hU, h2,
        henc]
      simp only [mat11, ternLaw, errLaw, tPoly]
end BgvSecurity
