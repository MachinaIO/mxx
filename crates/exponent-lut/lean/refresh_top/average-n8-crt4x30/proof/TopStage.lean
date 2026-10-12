import W_Claim

/-!
# Merged stages

A merged stage packs the per-instance protocol inputs of its parts into families and runs its body
at every instance (`∀ i : Fin 8, parallel_generatedRoot_K … i …`). `familyPack_get` reads a packed
family at an instance.
-/

namespace RefreshTopProof

open MxxRuntime

theorem familyPack_get {α : Type} {size : Nat} {count : Int} {values : List α}
    {output : Fin size → α} (h : familyPack count values output) (i : Fin size) :
    output i = values.get ⟨i.val, h.2.1 ▸ i.isLt⟩ := by
  obtain ⟨-, hl, hall⟩ := h
  obtain ⟨p, hp, he⟩ := hall i
  rw [he]
  congr 1
  exact Fin.ext hp

end RefreshTopProof
