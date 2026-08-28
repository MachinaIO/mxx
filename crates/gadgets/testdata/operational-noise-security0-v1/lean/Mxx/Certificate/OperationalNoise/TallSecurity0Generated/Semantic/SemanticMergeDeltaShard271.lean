import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge46175
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46175
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18176⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 33)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18176⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18176⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46175

namespace LeftMerge46177
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46177
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18176⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46176
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46176) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨18176⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨18176⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46177

namespace LeftMerge46178
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46178
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17091⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 29)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17091⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17091⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46178

namespace LeftMerge46180
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46180
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17091⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46179
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46179) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17091⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17091⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46180

namespace LeftMerge46181
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46181
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16804⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 28)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16804⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16804⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46181

namespace LeftMerge46183
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46183
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16804⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46182
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46182) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16804⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16804⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46183

namespace LeftMerge46184
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46184
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16685⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 27)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16685⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16685⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46184

namespace LeftMerge46186
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46186
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16685⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46185
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46185) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16685⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16685⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46186

namespace LeftMerge46187
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46187
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 34)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46187

namespace LeftMerge46189
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46189
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18211⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46188
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46188) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨18211⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨18211⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46189

namespace LeftMerge46190
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46190
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 32)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17910⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46190

namespace LeftMerge46192
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46192
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17910⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46191
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46191) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17910⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17910⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46192

namespace LeftMerge46193
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46193
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 30)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17126⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46193

namespace LeftMerge46195
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46195
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17126⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46194
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46194) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17126⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17126⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46195

namespace LeftMerge46196
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46196
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 26)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16314⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46196

namespace LeftMerge46198
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46198
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16314⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46197
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46197) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16314⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16314⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46198

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
