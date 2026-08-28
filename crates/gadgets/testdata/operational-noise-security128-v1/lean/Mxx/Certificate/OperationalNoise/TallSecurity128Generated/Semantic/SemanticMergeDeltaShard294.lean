import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge51413
def owner : Owner := ⟨.program ⟨257⟩, ⟨64320⟩⟩
def mergeEvent : Nat := 51413
def frameStart : Nat := 51347
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events200.exact51409RawTerms
def rightRaw : List Term := Proof.Events200.exact51407RawTerms
def group : MergeGroup := .operator 51409 51407
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51409) (leftOrdinal := 0)
    (rightResult := 51407) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51413

namespace LeftMerge51425
def owner : Owner := ⟨.program ⟨257⟩, ⟨65121⟩⟩
def mergeEvent : Nat := 51425
def frameStart : Nat := 51347
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩] } }
def leftRaw : List Term := Proof.Events200.exact51421RawTerms
def rightRaw : List Term := Proof.Events200.exact51398RawTerms
def group : MergeGroup := .operator 51421 51398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51421) (leftOrdinal := 0)
    (rightResult := 51398) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨65120⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51425

namespace LeftMerge51426
def owner : Owner := ⟨.program ⟨257⟩, ⟨65121⟩⟩
def mergeEvent : Nat := 51426
def frameStart : Nat := 51347
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩] } }
def leftRaw : List Term := Proof.Events200.exact51421RawTerms
def rightRaw : List Term := Proof.Events200.exact51398RawTerms
def group : MergeGroup := .operator 51421 51398
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51421) (leftOrdinal := 1)
    (rightResult := 51398) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨65120⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51426

namespace LeftMerge51428
def owner : Owner := ⟨.program ⟨257⟩, ⟨65121⟩⟩
def mergeEvent : Nat := 51428
def frameStart : Nat := 51347
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64153⟩⟩] } }
def rhsRaw : List Term := Proof.Events200.exact51395RawTerms
def group : MergeGroup := .relation 51427
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51427) (rhsResult := 51395)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨62872⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨65120⟩⟩) ⟨64153⟩ 51395) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨64153⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨62872⟩⟩], [⟨.program ⟨257⟩, ⟨64153⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51428

namespace LeftMerge51436
def owner : Owner := ⟨.program ⟨257⟩, ⟨63235⟩⟩
def mergeEvent : Nat := 51436
def frameStart : Nat := 51347
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨63233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events200.exact51409RawTerms
def rightRaw : List Term := Proof.Events200.exact51432RawTerms
def group : MergeGroup := .operator 51409 51432
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51409) (leftOrdinal := 0)
    (rightResult := 51432) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63233⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51436

namespace LeftMerge51453
def owner : Owner := ⟨.program ⟨257⟩, ⟨63839⟩⟩
def mergeEvent : Nat := 51453
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }
def rhsRaw : List Term := Proof.Events200.exact51450RawTerms
def group : MergeGroup := .relation 51452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51452) (rhsResult := 51450)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 51451 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63836⟩⟩]⟩) (none) 51450) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7214⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51453

namespace LeftMerge51454
def owner : Owner := ⟨.program ⟨257⟩, ⟨63839⟩⟩
def mergeEvent : Nat := 51454
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩] } }
def rhsRaw : List Term := Proof.Events200.exact51450RawTerms
def group : MergeGroup := .relation 51452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51452) (rhsResult := 51450)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 51451 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63836⟩⟩]⟩) (none) 51450) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51454

namespace LeftMerge51455
def owner : Owner := ⟨.program ⟨257⟩, ⟨63839⟩⟩
def mergeEvent : Nat := 51455
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64153⟩⟩] } }
def rhsRaw : List Term := Proof.Events200.exact51450RawTerms
def group : MergeGroup := .relation 51452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51452) (rhsResult := 51450)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 51451 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63836⟩⟩]⟩) (none) 51450) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64153⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨62872⟩⟩], [⟨.program ⟨257⟩, ⟨64153⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51455

namespace LeftMerge51456
def owner : Owner := ⟨.program ⟨257⟩, ⟨63839⟩⟩
def mergeEvent : Nat := 51456
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events200.exact51450RawTerms
def group : MergeGroup := .relation 51452
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 51452) (rhsResult := 51450)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63836⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 51451 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨63836⟩⟩]⟩) (none) 51450) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨63233⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨63233⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51456

namespace LeftMerge51461
def owner : Owner := ⟨.program ⟨257⟩, ⟨65123⟩⟩
def mergeEvent : Nat := 51461
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩] } }
def leftRaw : List Term := Proof.Events201.exact51457RawTerms
def rightRaw : List Term := Proof.Events200.exact51279RawTerms
def group : MergeGroup := .operator 51457 51279
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51457) (leftOrdinal := 0)
    (rightResult := 51279) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7187⟩⟩, ⟨.program ⟨257⟩, ⟨65120⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51461

namespace LeftMerge51462
def owner : Owner := ⟨.program ⟨257⟩, ⟨65123⟩⟩
def mergeEvent : Nat := 51462
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64153⟩⟩] } }
def leftRaw : List Term := Proof.Events201.exact51457RawTerms
def rightRaw : List Term := Proof.Events200.exact51279RawTerms
def group : MergeGroup := .operator 51457 51279
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51457) (leftOrdinal := 2)
    (rightResult := 51279) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64153⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨62872⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨64153⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨62872⟩⟩], [⟨.program ⟨257⟩, ⟨64153⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51462

namespace LeftMerge51488
def owner : Owner := ⟨.program ⟨257⟩, ⟨25347⟩⟩
def mergeEvent : Nat := 51488
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events007.exact1820RawTerms
def rightRaw : List Term := Proof.Events182.exact46653RawTerms
def group : MergeGroup := .operator 1820 46653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1820) (leftOrdinal := 0)
    (rightResult := 46653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25346⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25346⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51488

namespace LeftMerge51493
def owner : Owner := ⟨.program ⟨257⟩, ⟨11180⟩⟩
def mergeEvent : Nat := 51493
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def leftRaw : List Term := Proof.Events181.exact46523RawTerms
def rightRaw : List Term := Proof.Events086.exact22090RawTerms
def group : MergeGroup := .operator 46523 22090
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 46523) (leftOrdinal := 0)
    (rightResult := 22090) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51493

namespace LeftMerge51510
def owner : Owner := ⟨.program ⟨257⟩, ⟨59704⟩⟩
def mergeEvent : Nat := 51510
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25346⟩⟩, ⟨.program ⟨257⟩, ⟨59701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events201.exact51504RawTerms
def rightRaw : List Term := Proof.Events007.exact1823RawTerms
def group : MergeGroup := .operator 51504 1823
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51504) (leftOrdinal := 1)
    (rightResult := 1823) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25346⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59701⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨25346⟩⟩, ⟨.program ⟨257⟩, ⟨59701⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge51510

namespace LeftMerge51511
def owner : Owner := ⟨.program ⟨257⟩, ⟨59704⟩⟩
def mergeEvent : Nat := 51511
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨59701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }
def leftRaw : List Term := Proof.Events201.exact51504RawTerms
def rightRaw : List Term := Proof.Events007.exact1823RawTerms
def group : MergeGroup := .operator 51504 1823
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 51504) (leftOrdinal := 0)
    (rightResult := 1823) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7274⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59701⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨59701⟩⟩], [⟨.program ⟨257⟩, ⟨7274⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51511

namespace LeftMerge51516
def owner : Owner := ⟨.program ⟨257⟩, ⟨59705⟩⟩
def mergeEvent : Nat := 51516
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨59701⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events007.exact1823RawTerms
def rightRaw : List Term := Proof.Events182.exact46653RawTerms
def group : MergeGroup := .operator 1823 46653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1823) (leftOrdinal := 0)
    (rightResult := 46653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨59701⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11544⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11544⟩⟩, ⟨.program ⟨257⟩, ⟨59701⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge51516

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
