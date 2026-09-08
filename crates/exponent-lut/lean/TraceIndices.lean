import TraceIndicesData

namespace TraceIndices

private theorem residue_positions_valid : ∀ i, residuePositions i < width := by decide
private theorem inverse_positions_valid : ∀ i, inversePositions i < width := by decide

def residuePosition (i : Fin width) : Fin width :=
  ⟨residuePositions i, residue_positions_valid i⟩
def inversePosition (i : Fin width) : Fin width :=
  ⟨inversePositions i, inverse_positions_valid i⟩

def permutation : Fin width ≃ Fin width where
  toFun := residuePosition
  invFun := inversePosition
  left_inv := by decide
  right_inv := by decide

theorem dimension : ringDimension = stride * width := by decide
theorem width_power : width = 2 ^ widthLog := by decide
theorem all_odd : ∀ i, Odd (indices i) := by decide
theorem indices_valid : ∀ i, 0 < indices i ∧ indices i < 2 * ringDimension := by decide
theorem index_correspondence : ∀ i, indices i % (2 * width) =
    2 * (permutation i).val + 1 := by decide
theorem identity_index : indices ⟨0, by decide⟩ = 1 := by decide

end TraceIndices

#print axioms TraceIndices.index_correspondence
