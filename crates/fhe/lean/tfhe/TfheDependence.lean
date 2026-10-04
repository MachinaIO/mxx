import TfheNand

/-!
Which tape keys each quantity of the explicit model reads. The base keys (secrets, bootstrapping
key masks and encryption errors) are read throughout; blind-rotation step `i` reads the
bootstrapping key errors of steps before `i`, so the errors of step `i` are fresh when step `i`
multiplies them with its digits. Key switching reads everything but the key-switching errors.
-/

namespace MxxFheTfhe

open Mxx.Primitives MxxRuntime

irreducible_def lweKeys : Finset SampleKey := Finset.univ.image fun j : Fin lweN ↦ lweKey j
irreducible_def ringKeys : Finset SampleKey := Finset.univ.image fun k : Fin N ↦ ringKey k
irreducible_def maskKeys : Finset SampleKey :=
  Finset.univ.image fun x : Fin lweN × Fin 12 × Fin N ↦ bskMaskKey x.1 x.2.1 x.2.2
irreducible_def encKeys : Finset SampleKey := {encErrorKey 1, encErrorKey 2}
/-- The bootstrapping key errors of step `i`. -/
irreducible_def stepKeys (i : Nat) : Finset SampleKey :=
  Finset.univ.image fun x : Fin 12 × Fin N ↦ bskErrorKey i x.1 x.2
irreducible_def kskKeys : Finset SampleKey := Finset.univ.image fun e : Fin 8192 ↦ kskErrorKey e

/-- The bootstrapping key errors of the steps before `n`. -/
def errorKeys : Nat → Finset SampleKey
  | 0 => ∅
  | n + 1 => errorKeys n ∪ stepKeys n

/-- The keys read before blind-rotation step `n`: the base keys (secrets, bootstrapping key masks
and encryption errors) and the bootstrapping key errors of the earlier steps. -/
irreducible_def readKeys (n : Nat) : Finset SampleKey :=
  lweKeys ∪ ringKeys ∪ maskKeys ∪ encKeys ∪ errorKeys n

theorem mem_lweKeys (j : Fin lweN) : lweKey j ∈ lweKeys := by
  rw [lweKeys_def]; exact Finset.mem_image_of_mem _ (Finset.mem_univ j)
theorem mem_ringKeys (k : Fin N) : ringKey k ∈ ringKeys := by
  rw [ringKeys_def]; exact Finset.mem_image_of_mem _ (Finset.mem_univ k)
theorem mem_maskKeys (i : Fin lweN) (c : Fin 12) (k : Fin N) : bskMaskKey i c k ∈ maskKeys := by
  rw [maskKeys_def, Finset.mem_image]
  exact ⟨(i, c, k), by simp, rfl⟩
theorem mem_encKeys (stage : Nat) (hs : stage = 1 ∨ stage = 2) : encErrorKey stage ∈ encKeys := by
  rw [encKeys_def]; rcases hs with rfl | rfl <;> simp
theorem mem_stepKeys (i : Nat) (c : Fin 12) (k : Fin N) : bskErrorKey i c k ∈ stepKeys i := by
  rw [stepKeys_def, Finset.mem_image]
  exact ⟨(c, k), by simp, rfl⟩

theorem mem_stepKeys_iff {i : Nat} {k : SampleKey} :
    k ∈ stepKeys i ↔ ∃ (c : Fin 12) (x : Fin N), bskErrorKey i c x = k := by
  rw [stepKeys_def]
  simp only [Finset.mem_image, Finset.mem_univ, true_and, Prod.exists]

theorem mem_errorKeys_iff {n : Nat} {k : SampleKey} :
    k ∈ errorKeys n ↔ ∃ i < n, k ∈ stepKeys i := by
  induction n with
  | zero => simp [errorKeys]
  | succ n ih =>
    simp only [errorKeys, Finset.mem_union, ih]
    constructor
    · rintro (⟨i, hi, hk⟩ | hk)
      · exact ⟨i, by omega, hk⟩
      · exact ⟨n, by omega, hk⟩
    · rintro ⟨i, hi, hk⟩
      rcases Nat.lt_succ_iff_lt_or_eq.mp hi with hi | rfl
      · exact Or.inl ⟨i, hi, hk⟩
      · exact Or.inr hk

theorem mem_readKeys {n : Nat} {k : SampleKey} :
    k ∈ readKeys n ↔ k ∈ lweKeys ∨ k ∈ ringKeys ∨ k ∈ maskKeys ∨ k ∈ encKeys ∨
      ∃ i < n, k ∈ stepKeys i := by
  rw [readKeys_def]
  simp only [Finset.mem_union, mem_errorKeys_iff]
  tauto

theorem readKeys_mono {m n : Nat} (h : m ≤ n) : readKeys m ⊆ readKeys n := fun k hk ↦ by
  rw [mem_readKeys] at hk ⊢
  rcases hk with hk | hk | hk | hk | ⟨i, hi, hk⟩
  · exact Or.inl hk
  · exact Or.inr (Or.inl hk)
  · exact Or.inr (Or.inr (Or.inl hk))
  · exact Or.inr (Or.inr (Or.inr (Or.inl hk)))
  · exact Or.inr (Or.inr (Or.inr (Or.inr ⟨i, by omega, hk⟩)))

theorem lweKey_mem_read (n : Nat) (j : Fin lweN) : lweKey j ∈ readKeys n :=
  mem_readKeys.mpr (Or.inl (mem_lweKeys j))
theorem ringKey_mem_read (n : Nat) (k : Fin N) : ringKey k ∈ readKeys n :=
  mem_readKeys.mpr (Or.inr (Or.inl (mem_ringKeys k)))
theorem maskKey_mem_read (n : Nat) (i : Fin lweN) (c : Fin 12) (k : Fin N) :
    bskMaskKey i c k ∈ readKeys n :=
  mem_readKeys.mpr (Or.inr (Or.inr (Or.inl (mem_maskKeys i c k))))
theorem encKey_mem_read (n : Nat) (stage : Nat) (hs : stage = 1 ∨ stage = 2) :
    encErrorKey stage ∈ readKeys n :=
  mem_readKeys.mpr (Or.inr (Or.inr (Or.inr (Or.inl (mem_encKeys stage hs)))))
theorem errorKey_mem_read {i n : Nat} (hi : i < n) (c : Fin 12) (k : Fin N) :
    bskErrorKey i c k ∈ readKeys n :=
  mem_readKeys.mpr (Or.inr (Or.inr (Or.inr (Or.inr ⟨i, hi, mem_stepKeys i c k⟩))))

def Agree (S : Finset SampleKey) (t t' : SampleTape) : Prop := ∀ k ∈ S, t k = t' k

/-! ## Congruence of the model under agreeing tapes -/

section Congr

variable {t t' : SampleTape} {n : Nat}

theorem lweSecret_congr (h : Agree (readKeys n) t t') : lweSecret t = lweSecret t' :=
  funext fun j ↦ h _ (lweKey_mem_read n j)

theorem ringSecret_congr (h : Agree (readKeys n) t t') : ringSecret t = ringSecret t' := by
  unfold ringSecret
  congr 1
  funext k
  exact h _ (ringKey_mem_read n k)

theorem bskMask_congr (h : Agree (readKeys n) t t') (i : Fin lweN) : bskMask t i = bskMask t' i :=
  tapeMatrix_congr fun r c k ↦ by
    rw [Subsingleton.elim r 0]
    exact h (bskMaskKey i c k) (maskKey_mem_read n i c k)

theorem bskError_congr {i : Fin lweN} (h : Agree (readKeys n) t t') (hi : i.val < n) :
    bskError t i = bskError t' i := by
  funext r c
  unfold bskError
  congr 1
  funext k
  exact h (bskErrorKey i c k) (errorKey_mem_read hi c k)

theorem initialExp_congr (W : World) (h : Agree (readKeys n) t t') :
    initialExp W t = initialExp W t' := by
  have hs := lweSecret_congr h
  have h1 : encError t 1 = encError t' 1 := h _ (encKey_mem_read n 1 (by simp))
  have h2 : encError t 2 = encError t' 2 := h _ (encKey_mem_read n 2 (by simp))
  unfold initialExp combinedBody body
  rw [hs, h1, h2]

theorem gswA_congr (h : Agree (readKeys n) t t') (i : Fin lweN) : gswA t i = gswA t' i := by
  unfold gswA
  rw [lweSecret_congr h, bskMask_congr h i]

theorem gswB_congr {i : Fin lweN} (h : Agree (readKeys n) t t') (hi : i.val < n) :
    gswB t i = gswB t' i := by
  unfold gswB
  rw [lweSecret_congr h, ringSecret_congr h, bskMask_congr h i, bskError_congr h hi]

theorem accSeq_congr (W : World) (hn : n ≤ lweN) (h : Agree (readKeys n) t t') :
    accSeq W t n = accSeq W t' n := by
  induction n with
  | zero =>
    show accInit W t = accInit W t'
    unfold accInit
    rw [initialExp_congr W h]
  | succ n ih =>
    have hlt : n < lweN := by omega
    have hprev := ih (by omega) fun k hk ↦ h k (readKeys_mono (by omega) hk)
    show (if h : n < lweN then accStep W t ⟨n, h⟩ (accSeq W t n) else accSeq W t n) =
      (if h : n < lweN then accStep W t' ⟨n, h⟩ (accSeq W t' n) else accSeq W t' n)
    rw [dif_pos hlt, dif_pos hlt, hprev]
    unfold accStep bskRows
    rw [gswA_congr h, gswB_congr h (by simp)]

theorem stepDigits_congr (W : World) (i : Fin lweN) (h : Agree (readKeys i) t t') :
    stepDigits W t i = stepDigits W t' i := by
  unfold stepDigits
  rw [accSeq_congr W i.isLt.le h]

theorem stepShift_congr (W : World) (h : Agree (readKeys n) t t') (i : Fin lweN) :
    stepShift W t i = stepShift W t' i := by
  unfold stepShift
  rw [lweSecret_congr h]

theorem ksDigit_congr (W : World) (h : Agree (readKeys lweN) t t') (e : Fin 8192) :
    ksDigit W t e = ksDigit W t' e := by
  unfold ksDigit accMaskValues
  rw [accSeq_congr W le_rfl h]

end Congr

end MxxFheTfhe
