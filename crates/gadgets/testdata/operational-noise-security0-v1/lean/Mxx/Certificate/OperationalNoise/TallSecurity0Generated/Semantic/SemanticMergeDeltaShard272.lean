import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge46199
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46199
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 35)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46199

namespace LeftMerge46201
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46201
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18366⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46200
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46200) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨18366⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨18366⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46201

namespace LeftMerge46202
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46202
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 25)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16111⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46202

namespace LeftMerge46204
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46204
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16111⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46203
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46203) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16111⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16111⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46204

namespace LeftMerge46205
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46205
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 24)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15992⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46205

namespace LeftMerge46207
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46207
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15992⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46206
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46206) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15992⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15992⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46207

namespace LeftMerge46208
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46208
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 23)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46208

namespace LeftMerge46210
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46210
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15873⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46209
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46209) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15873⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15873⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46210

namespace LeftMerge46211
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46211
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 22)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46211

namespace LeftMerge46213
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46213
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15754⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46212
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46212) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15754⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46213

namespace LeftMerge46214
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46214
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 21)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46214

namespace LeftMerge46216
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46216
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15635⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46215
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46215) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15635⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46216

namespace LeftMerge46217
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46217
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 31)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46217

namespace LeftMerge46219
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46219
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨17345⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46218
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46218) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨17345⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46219

namespace LeftMerge46220
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46220
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩] } }
def leftRaw : List Term := Proof.Events180.exact46153RawTerms
def rightRaw : List Term := Proof.Events179.exact45994RawTerms
def group : MergeGroup := .operator 46153 45994
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46153) (leftOrdinal := 20)
    (rightResult := 45994) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18687⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46220

namespace LeftMerge46222
def owner : Owner := ⟨.program ⟨214⟩, ⟨18688⟩⟩
def mergeEvent : Nat := 46222
def frameStart : Nat := 45478
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨15374⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }
def rhsRaw : List Term := Proof.Events179.exact45991RawTerms
def group : MergeGroup := .relation 46221
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 46221) (rhsResult := 45991)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨18687⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨18687⟩⟩) ⟨18622⟩ 45991) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨18622⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨15374⟩⟩], [⟨.program ⟨214⟩, ⟨18622⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge46222

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
