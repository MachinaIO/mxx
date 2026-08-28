import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge204248
def owner : Owner := ⟨.program ⟨257⟩, ⟨39356⟩⟩
def mergeEvent : Nat := 204248
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37444⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38598⟩⟩] } }
def leftRaw : List Term := Proof.Events797.exact204243RawTerms
def rightRaw : List Term := Proof.Events797.exact204065RawTerms
def group : MergeGroup := .operator 204243 204065
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204243) (leftOrdinal := 2)
    (rightResult := 204065) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37444⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38598⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37444⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨38598⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37444⟩⟩], [⟨.program ⟨257⟩, ⟨38598⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204248

namespace LeftMerge204256
def owner : Owner := ⟨.program ⟨257⟩, ⟨39357⟩⟩
def mergeEvent : Nat := 204256
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩] } }
def leftRaw : List Term := Proof.Events797.exact204250RawTerms
def rightRaw : List Term := Proof.Events061.exact15622RawTerms
def group : MergeGroup := .operator 204250 15622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204250) (leftOrdinal := 0)
    (rightResult := 15622) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7223⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7161⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7223⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204256

namespace LeftMerge204257
def owner : Owner := ⟨.program ⟨257⟩, ⟨39357⟩⟩
def mergeEvent : Nat := 204257
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩] } }
def leftRaw : List Term := Proof.Events797.exact204250RawTerms
def rightRaw : List Term := Proof.Events061.exact15622RawTerms
def group : MergeGroup := .operator 204250 15622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204250) (leftOrdinal := 1)
    (rightResult := 15622) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7161⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204257

namespace LeftMerge204259
def owner : Owner := ⟨.program ⟨257⟩, ⟨39357⟩⟩
def mergeEvent : Nat := 204259
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events060.exact15615RawTerms
def group : MergeGroup := .relation 204258
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204258) (rhsResult := 15615)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7161⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7161⟩⟩) ⟨7046⟩ 15615) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6838⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨6838⟩⟩, ⟨.program ⟨257⟩, ⟨37665⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204259

namespace LeftMerge204273
def owner : Owner := ⟨.program ⟨257⟩, ⟨36675⟩⟩
def mergeEvent : Nat := 204273
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36673⟩⟩] } }
def leftRaw : List Term := Proof.Events764.exact195591RawTerms
def rightRaw : List Term := Proof.Events797.exact204267RawTerms
def group : MergeGroup := .operator 195591 204267
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195591) (leftOrdinal := 0)
    (rightResult := 204267) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36673⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36673⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204273

namespace LeftMerge204274
def owner : Owner := ⟨.program ⟨257⟩, ⟨36675⟩⟩
def mergeEvent : Nat := 204274
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36673⟩⟩] } }
def leftRaw : List Term := Proof.Events764.exact195591RawTerms
def rightRaw : List Term := Proof.Events797.exact204267RawTerms
def group : MergeGroup := .operator 195591 204267
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195591) (leftOrdinal := 1)
    (rightResult := 204267) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36673⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36673⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204274

namespace LeftMerge204276
def owner : Owner := ⟨.program ⟨257⟩, ⟨36675⟩⟩
def mergeEvent : Nat := 204276
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35918⟩⟩] } }
def rhsRaw : List Term := Proof.Events797.exact204264RawTerms
def group : MergeGroup := .relation 204275
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204275) (rhsResult := 204264)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36673⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36673⟩⟩) ⟨35918⟩ 204264) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35918⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨35918⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204276

namespace LeftMerge204290
def owner : Owner := ⟨.program ⟨257⟩, ⟨35535⟩⟩
def mergeEvent : Nat := 204290
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35532⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events797.exact204284RawTerms
def group : MergeGroup := .operator 192995 204284
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 204284) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35532⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35532⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204290

namespace LeftMerge204411
def owner : Owner := ⟨.program ⟨257⟩, ⟨36116⟩⟩
def mergeEvent : Nat := 204411
def frameStart : Nat := 204345
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events798.exact204407RawTerms
def rightRaw : List Term := Proof.Events798.exact204405RawTerms
def group : MergeGroup := .operator 204407 204405
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204407) (leftOrdinal := 0)
    (rightResult := 204405) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204411

namespace LeftMerge204423
def owner : Owner := ⟨.program ⟨257⟩, ⟨36674⟩⟩
def mergeEvent : Nat := 204423
def frameStart : Nat := 204345
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36673⟩⟩] } }
def leftRaw : List Term := Proof.Events798.exact204419RawTerms
def rightRaw : List Term := Proof.Events798.exact204396RawTerms
def group : MergeGroup := .operator 204419 204396
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204419) (leftOrdinal := 0)
    (rightResult := 204396) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36673⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36673⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204423

namespace LeftMerge204424
def owner : Owner := ⟨.program ⟨257⟩, ⟨36674⟩⟩
def mergeEvent : Nat := 204424
def frameStart : Nat := 204345
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36673⟩⟩] } }
def leftRaw : List Term := Proof.Events798.exact204419RawTerms
def rightRaw : List Term := Proof.Events798.exact204396RawTerms
def group : MergeGroup := .operator 204419 204396
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204419) (leftOrdinal := 1)
    (rightResult := 204396) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36673⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36673⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204424

namespace LeftMerge204426
def owner : Owner := ⟨.program ⟨257⟩, ⟨36674⟩⟩
def mergeEvent : Nat := 204426
def frameStart : Nat := 204345
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35918⟩⟩] } }
def rhsRaw : List Term := Proof.Events798.exact204393RawTerms
def group : MergeGroup := .relation 204425
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204425) (rhsResult := 204393)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36673⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36673⟩⟩) ⟨35918⟩ 204393) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35918⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨35918⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204426

namespace LeftMerge204434
def owner : Owner := ⟨.program ⟨257⟩, ⟨34987⟩⟩
def mergeEvent : Nat := 204434
def frameStart : Nat := 204345
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34985⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events798.exact204407RawTerms
def rightRaw : List Term := Proof.Events798.exact204430RawTerms
def group : MergeGroup := .operator 204407 204430
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 204407) (leftOrdinal := 0)
    (rightResult := 204430) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34985⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34985⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204434

namespace LeftMerge204451
def owner : Owner := ⟨.program ⟨257⟩, ⟨35535⟩⟩
def mergeEvent : Nat := 204451
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩] } }
def rhsRaw : List Term := Proof.Events798.exact204448RawTerms
def group : MergeGroup := .relation 204450
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204450) (rhsResult := 204448)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35532⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 204449 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35532⟩⟩]⟩) (none) 204448) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204451

namespace LeftMerge204452
def owner : Owner := ⟨.program ⟨257⟩, ⟨35535⟩⟩
def mergeEvent : Nat := 204452
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36673⟩⟩] } }
def rhsRaw : List Term := Proof.Events798.exact204448RawTerms
def group : MergeGroup := .relation 204450
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204450) (rhsResult := 204448)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35532⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 204449 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35532⟩⟩]⟩) (none) 204448) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36673⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36673⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge204452

namespace LeftMerge204453
def owner : Owner := ⟨.program ⟨257⟩, ⟨35535⟩⟩
def mergeEvent : Nat := 204453
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35918⟩⟩] } }
def rhsRaw : List Term := Proof.Events798.exact204448RawTerms
def group : MergeGroup := .relation 204450
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 204450) (rhsResult := 204448)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35532⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 204449 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35532⟩⟩]⟩) (none) 204448) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35918⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨35918⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge204453

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
