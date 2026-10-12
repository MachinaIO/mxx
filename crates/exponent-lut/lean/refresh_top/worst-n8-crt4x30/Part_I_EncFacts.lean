import TopInst
import Part_I_Enc
import TopLabel

/-! The label encodings of part `I` (written by `generator/gen_enc_facts.py`). -/

namespace RefreshTopProof.Part_I

open Mxx.Primitives MxxRuntime MxxExponentLut RefreshTopProof


/-- The label encodings of the encryption parallel `42`. -/
theorem encLabelRel42 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params}
    {A00 A01 A10 A11 : Fin 5 → ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1}
    {c e : Fin 5 → Int} {o0 o1 o2 o3 o4 o5 : Fin 5 → ExactMatrix Q6 N 1 12} {o6 : Fin 5 → ExactMatrix Q6 N 1 1}
    (h : ∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_42 Backend.backend model params i0 i
      (A00 i, A01 i, A10 i, A11 i, smM, oneM, c i, e i, ()) (o0 i, o1 i, o2 i, o3 i, o4 i, o5 i, o6 i, ()))
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    (hc : ∀ k, c k * c k = 1) (hr : ∀ k, e k * 2 + (1 - c k) * 8 = (labelExp k : Int) * 4)
    (he : ∀ k, 0 ≤ e k ∧ e k ≤ 7) :
    ∀ k : Fin 5, Rel sm ![![o0 k, o1 k], ![o2 k, o3 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • G) ![o4 k, o5 k] 26 := by
  intro k
  obtain ⟨hk, hrel, -⟩ := encLabel42 (h := h k) hsm hone hG
  simp only [Prod.mk.injEq] at hk
  obtain ⟨k0, k1, k2, k3⟩ := hk
  rw [label_message (hc k) (hr k) (he k).1 (he k).2, Int.toNat_natCast] at hrel
  rw [k0, k1, k2, k3]
  exact hrel


/-- The label encodings of the encryption parallel `85`. -/
theorem encLabelRel85 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params}
    {A00 A01 A10 A11 : Fin 5 → ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1}
    {c e : Fin 5 → Int} {o0 o1 o2 o3 o4 o5 : Fin 5 → ExactMatrix Q6 N 1 12} {o6 : Fin 5 → ExactMatrix Q6 N 1 1}
    (h : ∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_85 Backend.backend model params i0 i
      (A00 i, A01 i, A10 i, A11 i, smM, oneM, c i, e i, ()) (o0 i, o1 i, o2 i, o3 i, o4 i, o5 i, o6 i, ()))
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    (hc : ∀ k, c k * c k = 1) (hr : ∀ k, e k * 2 + (1 - c k) * 8 = (labelExp k : Int) * 4)
    (he : ∀ k, 0 ≤ e k ∧ e k ≤ 7) :
    ∀ k : Fin 5, Rel sm ![![o0 k, o1 k], ![o2 k, o3 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • G) ![o4 k, o5 k] 26 := by
  intro k
  obtain ⟨hk, hrel, -⟩ := encLabel85 (h := h k) hsm hone hG
  simp only [Prod.mk.injEq] at hk
  obtain ⟨k0, k1, k2, k3⟩ := hk
  rw [label_message (hc k) (hr k) (he k).1 (he k).2, Int.toNat_natCast] at hrel
  rw [k0, k1, k2, k3]
  exact hrel


/-- The label encodings of the encryption parallel `128`. -/
theorem encLabelRel128 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params}
    {A00 A01 A10 A11 : Fin 5 → ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1}
    {c e : Fin 5 → Int} {o0 o1 o2 o3 o4 o5 : Fin 5 → ExactMatrix Q6 N 1 12} {o6 : Fin 5 → ExactMatrix Q6 N 1 1}
    (h : ∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_128 Backend.backend model params i0 i
      (A00 i, A01 i, A10 i, A11 i, smM, oneM, c i, e i, ()) (o0 i, o1 i, o2 i, o3 i, o4 i, o5 i, o6 i, ()))
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    (hc : ∀ k, c k * c k = 1) (hr : ∀ k, e k * 2 + (1 - c k) * 8 = (labelExp k : Int) * 4)
    (he : ∀ k, 0 ≤ e k ∧ e k ≤ 7) :
    ∀ k : Fin 5, Rel sm ![![o0 k, o1 k], ![o2 k, o3 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • G) ![o4 k, o5 k] 26 := by
  intro k
  obtain ⟨hk, hrel, -⟩ := encLabel128 (h := h k) hsm hone hG
  simp only [Prod.mk.injEq] at hk
  obtain ⟨k0, k1, k2, k3⟩ := hk
  rw [label_message (hc k) (hr k) (he k).1 (he k).2, Int.toNat_natCast] at hrel
  rw [k0, k1, k2, k3]
  exact hrel


/-- The label encodings of the encryption parallel `171`. -/
theorem encLabelRel171 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params}
    {A00 A01 A10 A11 : Fin 5 → ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1}
    {c e : Fin 5 → Int} {o0 o1 o2 o3 o4 o5 : Fin 5 → ExactMatrix Q6 N 1 12} {o6 : Fin 5 → ExactMatrix Q6 N 1 1}
    (h : ∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_171 Backend.backend model params i0 i
      (A00 i, A01 i, A10 i, A11 i, smM, oneM, c i, e i, ()) (o0 i, o1 i, o2 i, o3 i, o4 i, o5 i, o6 i, ()))
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    (hc : ∀ k, c k * c k = 1) (hr : ∀ k, e k * 2 + (1 - c k) * 8 = (labelExp k : Int) * 4)
    (he : ∀ k, 0 ≤ e k ∧ e k ≤ 7) :
    ∀ k : Fin 5, Rel sm ![![o0 k, o1 k], ![o2 k, o3 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • G) ![o4 k, o5 k] 26 := by
  intro k
  obtain ⟨hk, hrel, -⟩ := encLabel171 (h := h k) hsm hone hG
  simp only [Prod.mk.injEq] at hk
  obtain ⟨k0, k1, k2, k3⟩ := hk
  rw [label_message (hc k) (hr k) (he k).1 (he k).2, Int.toNat_natCast] at hrel
  rw [k0, k1, k2, k3]
  exact hrel


/-- The label encodings of the encryption parallel `214`. -/
theorem encLabelRel214 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params}
    {A00 A01 A10 A11 : Fin 5 → ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1}
    {c e : Fin 5 → Int} {o0 o1 o2 o3 o4 o5 : Fin 5 → ExactMatrix Q6 N 1 12} {o6 : Fin 5 → ExactMatrix Q6 N 1 1}
    (h : ∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_214 Backend.backend model params i0 i
      (A00 i, A01 i, A10 i, A11 i, smM, oneM, c i, e i, ()) (o0 i, o1 i, o2 i, o3 i, o4 i, o5 i, o6 i, ()))
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    (hc : ∀ k, c k * c k = 1) (hr : ∀ k, e k * 2 + (1 - c k) * 8 = (labelExp k : Int) * 4)
    (he : ∀ k, 0 ≤ e k ∧ e k ≤ 7) :
    ∀ k : Fin 5, Rel sm ![![o0 k, o1 k], ![o2 k, o3 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • G) ![o4 k, o5 k] 26 := by
  intro k
  obtain ⟨hk, hrel, -⟩ := encLabel214 (h := h k) hsm hone hG
  simp only [Prod.mk.injEq] at hk
  obtain ⟨k0, k1, k2, k3⟩ := hk
  rw [label_message (hc k) (hr k) (he k).1 (he k).2, Int.toNat_natCast] at hrel
  rw [k0, k1, k2, k3]
  exact hrel


/-- The label encodings of the encryption parallel `257`. -/
theorem encLabelRel257 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params}
    {A00 A01 A10 A11 : Fin 5 → ExactMatrix Q6 N 1 12} {smM oneM : ExactMatrix Q6 N 1 1}
    {c e : Fin 5 → Int} {o0 o1 o2 o3 o4 o5 : Fin 5 → ExactMatrix Q6 N 1 12} {o6 : Fin 5 → ExactMatrix Q6 N 1 1}
    (h : ∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_257 Backend.backend model params i0 i
      (A00 i, A01 i, A10 i, A11 i, smM, oneM, c i, e i, ()) (o0 i, o1 i, o2 i, o3 i, o4 i, o5 i, o6 i, ()))
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q6 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q6 N 1 12} (hG : gadgetMatrixRuns Backend.backend 32768 12 G)
    (hc : ∀ k, c k * c k = 1) (hr : ∀ k, e k * 2 + (1 - c k) * 8 = (labelExp k : Int) * 4)
    (he : ∀ k, 0 ≤ e k ∧ e k ≤ 7) :
    ∀ k : Fin 5, Rel sm ![![o0 k, o1 k], ![o2 k, o3 k]]
      (fun j ↦ reducePoly Q6 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • G) ![o4 k, o5 k] 26 := by
  intro k
  obtain ⟨hk, hrel, -⟩ := encLabel257 (h := h k) hsm hone hG
  simp only [Prod.mk.injEq] at hk
  obtain ⟨k0, k1, k2, k3⟩ := hk
  rw [label_message (hc k) (hr k) (he k).1 (he k).2, Int.toNat_natCast] at hrel
  rw [k0, k1, k2, k3]
  exact hrel


/-- The label encodings of the encryption parallel `331`. -/
theorem encLabelRel331 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params}
    {A00 A01 A10 A11 : Fin 5 → ExactMatrix Q7 N 1 14} {smM oneM : ExactMatrix Q7 N 1 1}
    {c e : Fin 5 → Int} {o0 o1 o2 o3 o4 o5 : Fin 5 → ExactMatrix Q7 N 1 14} {o6 : Fin 5 → ExactMatrix Q7 N 1 1}
    (h : ∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_331 Backend.backend model params i0 i
      (A00 i, A01 i, A10 i, A11 i, smM, oneM, c i, e i, ()) (o0 i, o1 i, o2 i, o3 i, o4 i, o5 i, o6 i, ()))
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q7 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 32768 14 G)
    (hc : ∀ k, c k * c k = 1) (hr : ∀ k, e k * 2 + (1 - c k) * 8 = (labelExp k : Int) * 4)
    (he : ∀ k, 0 ≤ e k ∧ e k ≤ 7) :
    ∀ k : Fin 5, Rel sm ![![o0 k, o1 k], ![o2 k, o3 k]]
      (fun j ↦ reducePoly Q7 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • G) ![o4 k, o5 k] 26 := by
  intro k
  obtain ⟨hk, hrel, -⟩ := encLabel331 (h := h k) hsm hone hG
  simp only [Prod.mk.injEq] at hk
  obtain ⟨k0, k1, k2, k3⟩ := hk
  rw [label_message (hc k) (hr k) (he k).1 (he k).2, Int.toNat_natCast] at hrel
  rw [k0, k1, k2, k3]
  exact hrel


/-- The label encodings of the encryption parallel `375`. -/
theorem encLabelRel375 {i0 : Nat} [Fact (i0 < 8)] {model : HashModel} {params : Stage_encrypt.Params}
    {A00 A01 A10 A11 : Fin 5 → ExactMatrix Q7 N 1 14} {smM oneM : ExactMatrix Q7 N 1 1}
    {c e : Fin 5 → Int} {o0 o1 o2 o3 o4 o5 : Fin 5 → ExactMatrix Q7 N 1 14} {o6 : Fin 5 → ExactMatrix Q7 N 1 1}
    (h : ∀ i : Fin 5, Stage_encrypt.parallel_parallel_generatedRoot_454_375 Backend.backend model params i0 i
      (A00 i, A01 i, A10 i, A11 i, smM, oneM, c i, e i, ()) (o0 i, o1 i, o2 i, o3 i, o4 i, o5 i, o6 i, ()))
    {sm : ErrorPoly N} (hsm : smM = fun _ _ ↦ reducePoly Q7 N sm) (hone : oneM = 1)
    {G : ExactMatrix Q7 N 1 14} (hG : gadgetMatrixRuns Backend.backend 32768 14 G)
    (hc : ∀ k, c k * c k = 1) (hr : ∀ k, e k * 2 + (1 - c k) * 8 = (labelExp k : Int) * 4)
    (he : ∀ k, 0 ≤ e k ∧ e k ≤ 7) :
    ∀ k : Fin 5, Rel sm ![![o0 k, o1 k], ![o2 k, o3 k]]
      (fun j ↦ reducePoly Q7 N (Xe (n := N) (2 * labelExp k) * unitPair sm j) • G) ![o4 k, o5 k] 26 := by
  intro k
  obtain ⟨hk, hrel, -⟩ := encLabel375 (h := h k) hsm hone hG
  simp only [Prod.mk.injEq] at hk
  obtain ⟨k0, k1, k2, k3⟩ := hk
  rw [label_message (hc k) (hr k) (he k).1 (he k).2, Int.toNat_natCast] at hrel
  rw [k0, k1, k2, k3]
  exact hrel


end RefreshTopProof.Part_I
