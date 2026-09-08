import NumericBoundsData

namespace ExponentGeneratedNumeric

/-- Kernel-checked arithmetic for every exported slot, including strict decoding
exactly when the candidate reports worst-case acceptance. -/
theorem all_slots_valid : slots.all (fun slot => decide (slot.Valid accepted)) = true := by
  decide

theorem slot_valid (slot : MxxExponentLut.RefreshNumericSlot) (member : slot ∈ slots) :
    slot.Valid accepted := by
  exact of_decide_eq_true (List.all_eq_true.mp all_slots_valid slot member)

end ExponentGeneratedNumeric

#print axioms ExponentGeneratedNumeric.slot_valid
