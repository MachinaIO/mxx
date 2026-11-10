import Reductions
import BgvNtt

/-!
# The generated BGV computations in closed form

Each generated computation draws its samplers and returns the outputs its relation determines.
These lemmas compute those outputs: the relations are functional, and the closed forms satisfy
them.
-/

namespace BgvSecurity

open Mxx.Primitives MxxRuntime MxxSecurity

abbrev Q : Nat := 5846006548020969210596774788483421161649837621249
abbrev QP : Nat := 421249166578543632464236954685046510773404157767471124412817604609

instance : Fact (1 < Q) := ⟨by decide⟩
instance : Fact (1 < QP) := ⟨by decide⟩

/-- `t` as a ring element of `R_q`. -/
noncomputable abbrev tPoly {q : Nat} : ExactPoly q n := ((1032193 : Int) : ExactPoly q n)

/-- The slot-permutation table of the encoding. -/
abbrev encodeTable : Nat := Sim_2_encrypt.generatedRoot.table_6

theorem encodeEntry_lt (i : Fin n) : packedEntry 13 encodeTable i.val < n :=
  (Nat.mod_lt _ (by decide)).trans_le (by decide)

theorem encodeEntry_lt_int (i : Fin n) : ((packedEntry 13 encodeTable i.val : Nat) : Int) < 8192 := by
  have h : packedEntry 13 encodeTable i.val < 8192 := encodeEntry_lt i
  exact Int.ofNat_lt.mpr h

theorem ext21 {q : Nat} {o o' : ExactMatrix q n 2 1} (h0 : o 0 0 = o' 0 0) (h1 : o 1 0 = o' 1 0) :
    o = o' := by
  funext r c
  fin_cases r <;> fin_cases c
  · exact h0
  · exact h1

theorem concatRows_of_rows {q : Nat} {l r : ExactMatrix q n 1 1} {o : ExactMatrix q n (1 + 1) 1}
    (h0 : o 0 0 = l 0 0) (h1 : o 1 0 = r 0 0) : concatRows l r o := by
  intro row column
  fin_cases row <;> fin_cases column
  · exact h0
  · exact h1

/-- The native evaluations of the slot vector `m`: slot `packedEntry i` at position `i`. -/
def nativeOf (m : Fin n → Int) : Fin n → Int := fun i ↦ m ⟨_, encodeEntry_lt i⟩

/-- The plaintext polynomial of `m` in `R_t`. -/
noncomputable def plainOf (m : Fin n → Int) : ExactMatrix t n 1 1 :=
  Classical.choose (fromValues_exists (nativeOf m))

theorem plainOf_spec (m : Fin n → Int) : polynomialFromValues true (nativeOf m) (plainOf m) :=
  Classical.choose_spec (fromValues_exists (nativeOf m))

/-- The encoded message, lifted to `R_Q`. -/
noncomputable def encodeSlots (m : Fin n → Int) : ExactMatrix Q n 1 1 :=
  centeredRebase (plainOf m)

theorem table_values (w : Fin n → Int)
    (h : polynomialValues false (packedPolynomial (q := t) (n := n) 13 8192 encodeTable) w) :
    ∀ i, w i = packedEntry 13 encodeTable i.val :=
  polynomialValues_packed (by decide) (by decide) h
    (fun j ↦ (Nat.mod_lt _ (by decide)).trans_le (by decide))

/-- The coefficient values of the packed table polynomial. -/
noncomputable def tableValues : Fin n → Int := fun i ↦
  (((packedPolynomial (q := t) (n := n) 13 8192 encodeTable 0 0).coeff i).val : Int)

theorem tableValues_holds :
    polynomialValues false (packedPolynomial (q := t) (n := n) 13 8192 encodeTable) tableValues := by
  simp only [polynomialValues, Bool.false_eq_true, if_false]
  intro i
  rfl

theorem tableValues_eq (i : Fin n) : tableValues i = packedEntry 13 encodeTable i.val :=
  table_values _ tableValues_holds i

/-- The `i`-th native evaluation is the slot the table names. -/
theorem slot_read {m : Fin n → Int} {w : Fin n → Int}
    (hw : ∀ i, w i = packedEntry 13 encodeTable i.val) (i : Fin n) {value : Int}
    (h : familyGetDynamic m (w i) value) : value = nativeOf m i := by
  obtain ⟨position, hposition, rfl⟩ := h
  congr 1
  exact Fin.ext (by have := hposition.trans (hw i); exact_mod_cast this)

theorem slot_read_holds (m : Fin n → Int) (i : Fin n) :
    familyGetDynamic m (tableValues i) (nativeOf m i) :=
  ⟨⟨_, encodeEntry_lt i⟩, by rw [tableValues_eq], rfl⟩

/-! ## Simulator of the ciphertext hop -/

/-- The public key the ciphertext simulator derives from the challenge row `a`. -/
noncomputable def pkOfRow (a : ExactMatrix Q n 1 2) : ExactMatrix Q n 2 1 :=
  fun r _ ↦ if r.val = 0 then tPoly * a 0 0 else tPoly * a 0 1

/-- The challenge ciphertext of the ciphertext simulator from the challenge row `b`. -/
noncomputable def ctOfRow (b : ExactMatrix Q n 1 2) (E : ExactMatrix Q n 1 1) :
    ExactMatrix Q n 2 1 :=
  fun r _ ↦ if r.val = 0 then tPoly * b 0 0 else tPoly * b 0 1 + E 0 0

theorem slice_col0 {q : Nat} (a : ExactMatrix q n 1 2) :
    sliceMatrix a 0 1 0 1 (fun _ _ ↦ a 0 0 : ExactMatrix q n 1 1) := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    fun row column _ _ ↦ ?_⟩
  fin_cases row; fin_cases column; rfl

theorem slice_col1 {q : Nat} (a : ExactMatrix q n 1 2) :
    sliceMatrix a 0 1 1 2 (fun _ _ ↦ a 0 1 : ExactMatrix q n 1 1) := by
  refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, by decide, by decide,
    fun row column _ _ ↦ ?_⟩
  fin_cases row; fin_cases column; rfl

theorem sim2_setup_eq (b a : ExactMatrix Q n 1 2) :
    Sim_2_setup.generatedRoot.comp GeneratedScheme.Sim_2_setup_params (b, a, ()) =
      (do let ek ← (sampleResidue : Comp (ExactMatrix QP n 2 3)); pure (b, ek, pkOfRow a, ())) := by
  unfold Sim_2_setup.generatedRoot.comp
  congr 1
  funext ek
  congr 1
  refine determined_eq ⟨⟨ek, fun _ _ ↦ a 0 0, fun _ _ ↦ a 0 1, pkOfRow a⟩, rfl, ?_⟩ ?_
  · refine ⟨rfl, by decide, by decide, by decide, by decide, by decide, by decide,
      slice_col0 a, by decide, by decide, by decide, by decide, by decide, by decide,
      slice_col1 a, fun row column ↦ ?_, rfl⟩
    fin_cases row <;> fin_cases column <;> rfl
  · rintro outputs ⟨w, hw, h⟩
    obtain ⟨-, -, -, -, -, -, -, hs0, -, -, -, -, -, -, hs1, hcat, rfl⟩ := h
    have h0 := sliceMatrix_entry hs0 (by decide) (by decide)
    have h1 := sliceMatrix_entry hs1 (by decide) (by decide)
    simp only [hw, Prod.mk.injEq, true_and, and_true]
    funext row column
    rw [hcat row column]
    fin_cases row <;> fin_cases column
    · simp only [matrixScale, pkOfRow, tPoly]
      simp only [Fin.zero_eta, Fin.isValue] at h0 ⊢
      rw [h0]
      rfl
    · simp only [matrixScale, pkOfRow, tPoly]
      rw [dif_neg (lt_irrefl 1), if_neg one_ne_zero]
      exact congrArg (fun x ↦ ((1032193 : Int) : ExactPoly Q n) * x) h1

theorem sim2_encrypt_eq (b : ExactMatrix Q n 1 2) (m : Fin n → Int) :
    Sim_2_encrypt.generatedRoot.comp GeneratedScheme.Sim_2_encrypt_params (b, m, ()) =
      pure (ctOfRow b (encodeSlots m)) := by
  unfold Sim_2_encrypt.generatedRoot.comp
  congr 1
  refine determined_eq ⟨⟨fun _ _ ↦ b 0 0, fun _ _ ↦ b 0 1, tableValues, nativeOf m, plainOf m,
    encodeSlots m, ctOfRow b (encodeSlots m)⟩, ?_⟩ ?_
  · dsimp only [Sim_2_encrypt.generatedRoot.body]
    refine ⟨by decide, by decide, by decide, by decide, by decide, by decide, slice_col0 b,
      by decide, by decide, by decide, by decide, by decide, by decide, slice_col1 b,
      tableValues_holds, rfl, fun i ↦ ⟨nativeOf m i, ?_, ?_, slot_read_holds m i, rfl⟩,
      plainOf_spec m, ⟨Fact.out, rfl⟩, fun row column ↦ ?_, rfl⟩
    · rw [tableValues_eq]; exact Int.natCast_nonneg _
    · rw [tableValues_eq]; exact encodeEntry_lt_int i
    · revert row column
      refine concatRows_of_rows ?_ ?_
      · simp [ctOfRow, matrixScale, tPoly]
      · simp [ctOfRow, matrixScale, matrixAdd, tPoly]
  · rintro outputs ⟨w, h⟩
    obtain ⟨-, -, -, -, -, -, hs0, -, -, -, -, -, -, hs1, hvals, -, hpar, hfrom, hreb, hcat, rfl⟩ := h
    have h0 := sliceMatrix_entry hs0 (by decide) (by decide)
    have h1 := sliceMatrix_entry hs1 (by decide) (by decide)
    have htab := table_values _ hvals
    have hnative : w.w_8_0 = nativeOf m := by
      funext i
      obtain ⟨value, -, -, hget, hout⟩ := hpar i
      rw [hout]
      exact slot_read htab i hget
    rw [hnative] at hfrom
    have hplain : w.w_9_0 = plainOf m := fromValues_unique hfrom (plainOf_spec m)
    have henc : w.w_10_0 = encodeSlots m := by rw [hreb.2, hplain]; rfl
    have h0' : w.w_1_0 0 0 = b 0 0 := h0
    have h1' : w.w_3_0 0 0 = b 0 1 := h1
    obtain ⟨c0, c1⟩ := concatRows_one_one hcat 0
    refine ext21 ?_ ?_
    · rw [c0]
      simp only [matrixScale, ctOfRow, tPoly, h0']
      rfl
    · rw [c1]
      simp [matrixScale, matrixAdd, ctOfRow, tPoly, h1', henc]

end BgvSecurity
