import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge188416
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188416
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨43038⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188415
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188415) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨43038⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨43038⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188416

namespace LeftMerge188417
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188417
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40358⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188386RawTerms
def rightRaw : List Term := Proof.Events735.exact188227RawTerms
def group : MergeGroup := .operator 188386 188227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188386) (leftOrdinal := 26)
    (rightResult := 188227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40358⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188417

namespace LeftMerge188419
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188419
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40358⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188418
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188418) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40358⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40358⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188419

namespace LeftMerge188420
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188420
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37682⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188386RawTerms
def rightRaw : List Term := Proof.Events735.exact188227RawTerms
def group : MergeGroup := .operator 188386 188227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188386) (leftOrdinal := 25)
    (rightResult := 188227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37682⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188420

namespace LeftMerge188422
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188422
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37682⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188421
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188421) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37682⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188422

namespace LeftMerge188423
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188423
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨35002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188386RawTerms
def rightRaw : List Term := Proof.Events735.exact188227RawTerms
def group : MergeGroup := .operator 188386 188227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188386) (leftOrdinal := 24)
    (rightResult := 188227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨35002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188423

namespace LeftMerge188425
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188425
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨35002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188424
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188424) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨35002⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188425

namespace LeftMerge188426
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188426
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188386RawTerms
def rightRaw : List Term := Proof.Events735.exact188227RawTerms
def group : MergeGroup := .operator 188386 188227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188386) (leftOrdinal := 22)
    (rightResult := 188227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188426

namespace LeftMerge188428
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188428
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188427
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188427) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29338⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188428

namespace LeftMerge188429
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188429
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26658⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188386RawTerms
def rightRaw : List Term := Proof.Events735.exact188227RawTerms
def group : MergeGroup := .operator 188386 188227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188386) (leftOrdinal := 21)
    (rightResult := 188227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26658⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188429

namespace LeftMerge188431
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188431
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26658⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188430
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188430) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26658⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188431

namespace LeftMerge188432
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188432
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66811⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188386RawTerms
def rightRaw : List Term := Proof.Events735.exact188227RawTerms
def group : MergeGroup := .operator 188386 188227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188386) (leftOrdinal := 35)
    (rightResult := 188227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66811⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188432

namespace LeftMerge188434
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188434
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66811⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188433
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188433) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨66811⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188434

namespace LeftMerge188435
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188435
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63138⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188386RawTerms
def rightRaw : List Term := Proof.Events735.exact188227RawTerms
def group : MergeGroup := .operator 188386 188227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188386) (leftOrdinal := 34)
    (rightResult := 188227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63138⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188435

namespace LeftMerge188437
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188437
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63138⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }
def rhsRaw : List Term := Proof.Events735.exact188224RawTerms
def group : MergeGroup := .relation 188436
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 188436) (rhsResult := 188224)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71329⟩⟩) ⟨68848⟩ 188224) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68848⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨63138⟩⟩], [⟨.program ⟨257⟩, ⟨68848⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188437

namespace LeftMerge188438
def owner : Owner := ⟨.program ⟨257⟩, ⟨71330⟩⟩
def mergeEvent : Nat := 188438
def frameStart : Nat := 187711
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨60158⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩] } }
def leftRaw : List Term := Proof.Events735.exact188386RawTerms
def rightRaw : List Term := Proof.Events735.exact188227RawTerms
def group : MergeGroup := .operator 188386 188227
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 188386) (leftOrdinal := 33)
    (rightResult := 188227) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨60158⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71329⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨60158⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71329⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge188438

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
