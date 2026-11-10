import TfheBridge

/-!
# Closed forms of the deterministic TFHE stages

The decryption and plaintext graphs determine their outputs; `decrypt_eq` and `plain_eq` give
them in closed form.
-/

set_option maxRecDepth 16384

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime MxxSecurity GeneratedScheme

/-! ## Small relation helpers -/

/-- `select` over two candidates indexed by a decided condition. -/
theorem select_two {a b output : Int} {c : Prop} [Decidable c]
    (h : select (if decide c then 1 else 0) [a, b] output) : output = if c then b else a := by
  obtain ⟨position, hposition, houtput⟩ := h
  rw [houtput]
  by_cases hc : c
  · simp only [hc, decide_true, if_true] at hposition ⊢
    have : position = 1 := Fin.ext (by simpa using hposition)
    subst this; rfl
  · simp only [hc, decide_false, Bool.false_eq_true, if_false] at hposition ⊢
    have : position = 0 := Fin.ext (by simpa using hposition)
    subst this; rfl

theorem select_two_holds {a b : Int} {c : Prop} [Decidable c] :
    select (if decide c then 1 else 0) [a, b] (if c then b else a) := by
  by_cases hc : c
  · exact ⟨1, by simp [hc], by simp [hc]⟩
  · exact ⟨0, by simp [hc], by simp [hc]⟩

theorem familyGet_holds {α : Type} {count : Nat} (family : Fin count → α) (i : Fin count) :
    familyGetDynamic family (i.val : Int) (family i) := ⟨i, rfl, rfl⟩

theorem familyGet_eq {α : Type} {count : Nat} {family : Fin count → α} {i : Fin count}
    {output : α} (h : familyGetDynamic family (i.val : Int) output) : output = family i := by
  obtain ⟨position, hposition, rfl⟩ := h
  congr 1
  exact Fin.ext (by exact_mod_cast hposition)

/-- The inner product `⟨a, s⟩` that decryption reads. -/
theorem dot_eq (a s : Fin lweN → Int) :
    intMatrixVectorProduct (outer := 1) false a s 0 = ∑ j : Fin lweN, a j * s j := by
  rw [intMatrixVectorProduct_rows (by rfl)]
  simp

/-! ## Decryption and the plaintext circuit -/

/-- The sign decoding of an LWE ciphertext `(a, b)` under `s`: `1` when the centered phase
`b - ⟨a, s⟩` is nonnegative. -/
def decOf (b : Int) (a s : Fin lweN → Int) : Int :=
  let phase := (b - ∑ j : Fin lweN, a j * s j) % 4294967296
  1 - if (if phase * 2 ≤ 4294967296 then phase else phase - 4294967296) < 0 then 1 else 0

theorem decrypt_eq (b : Int) (a s : Fin lweN → Int) :
    (determined fun outputs ↦ ∃ witness,
      Decrypt_1.generatedRoot.body Decrypt_1_params (b, a, s, ()) outputs witness) =
      decOf b a s := by
  let ph := (b - ∑ j : Fin lweN, a j * s j) % 4294967296
  refine determined_eq ⟨⟨∑ j : Fin lweN, a j * s j,
    if ph * 2 ≤ 4294967296 then ph else ph - 4294967296⟩, ?_⟩ ?_
  · refine ⟨?_, by decide, ?_, ?_, rfl, ?_, ?_⟩
    · exact ⟨0, rfl, (dot_eq a s).symm⟩
    · dsimp only; split <;> norm_num
    · dsimp only; split <;> norm_num
    · exact select_two_holds
    · simp only [decOf, decide_eq_true_eq]
      rfl
  · rintro other ⟨witness, hdot, _, _, _, _, hsel, hout⟩
    have h5 : witness.w_5_0 = ∑ j : Fin lweN, a j * s j :=
      (familyGet_eq (i := (0 : Fin 1)) hdot).trans (dot_eq a s)
    have h16 := select_two hsel
    rw [hout]
    simp only [decOf, decide_eq_true_eq]
    rw [h16, h5]

theorem plain_eq (m0 m1 : Int) :
    (determined fun outputs ↦ Plain.generatedRoot Plain_params (m0, m1, ()) outputs) =
      1 - m0 * m1 :=
  determined_eq ⟨⟨⟩, rfl⟩ fun _ ⟨_, h⟩ ↦ h

end MxxFheTfhe
