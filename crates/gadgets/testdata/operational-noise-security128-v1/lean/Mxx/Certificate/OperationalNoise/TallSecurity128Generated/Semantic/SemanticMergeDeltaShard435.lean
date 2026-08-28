import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge74345
def owner : Owner := ⟨.program ⟨257⟩, ⟨53164⟩⟩
def mergeEvent : Nat := 74345
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53162⟩⟩] } }
def leftRaw : List Term := Proof.Events264.exact67822RawTerms
def rightRaw : List Term := Proof.Events290.exact74338RawTerms
def group : MergeGroup := .operator 67822 74338
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 67822) (leftOrdinal := 1)
    (rightResult := 74338) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53162⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53162⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge74345

namespace LeftMerge74347
def owner : Owner := ⟨.program ⟨257⟩, ⟨53164⟩⟩
def mergeEvent : Nat := 74347
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52223⟩⟩] } }
def rhsRaw : List Term := Proof.Events290.exact74335RawTerms
def group : MergeGroup := .relation 74346
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 74346) (rhsResult := 74335)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53162⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53162⟩⟩) ⟨52223⟩ 74335) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52223⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨52223⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge74347

namespace LeftMerge74361
def owner : Owner := ⟨.program ⟨257⟩, ⟨51895⟩⟩
def mergeEvent : Nat := 74361
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51892⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events290.exact74355RawTerms
def group : MergeGroup := .operator 61370 74355
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 74355) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨51892⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51892⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge74361

namespace LeftMerge74482
def owner : Owner := ⟨.program ⟨257⟩, ⟨52396⟩⟩
def mergeEvent : Nat := 74482
def frameStart : Nat := 74416
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events290.exact74478RawTerms
def rightRaw : List Term := Proof.Events290.exact74476RawTerms
def group : MergeGroup := .operator 74478 74476
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 74478) (leftOrdinal := 0)
    (rightResult := 74476) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge74482

namespace LeftMerge74494
def owner : Owner := ⟨.program ⟨257⟩, ⟨53163⟩⟩
def mergeEvent : Nat := 74494
def frameStart : Nat := 74416
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53162⟩⟩] } }
def leftRaw : List Term := Proof.Events290.exact74490RawTerms
def rightRaw : List Term := Proof.Events290.exact74467RawTerms
def group : MergeGroup := .operator 74490 74467
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 74490) (leftOrdinal := 0)
    (rightResult := 74467) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53162⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53162⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge74494

namespace LeftMerge74495
def owner : Owner := ⟨.program ⟨257⟩, ⟨53163⟩⟩
def mergeEvent : Nat := 74495
def frameStart : Nat := 74416
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53162⟩⟩] } }
def leftRaw : List Term := Proof.Events290.exact74490RawTerms
def rightRaw : List Term := Proof.Events290.exact74467RawTerms
def group : MergeGroup := .operator 74490 74467
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 74490) (leftOrdinal := 1)
    (rightResult := 74467) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨53162⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53162⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge74495

namespace LeftMerge74497
def owner : Owner := ⟨.program ⟨257⟩, ⟨53163⟩⟩
def mergeEvent : Nat := 74497
def frameStart : Nat := 74416
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52223⟩⟩] } }
def rhsRaw : List Term := Proof.Events290.exact74464RawTerms
def group : MergeGroup := .relation 74496
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 74496) (rhsResult := 74464)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨53162⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨53162⟩⟩) ⟨52223⟩ 74464) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨52223⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨52223⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge74497

namespace LeftMerge74505
def owner : Owner := ⟨.program ⟨257⟩, ⟨51301⟩⟩
def mergeEvent : Nat := 74505
def frameStart : Nat := 74416
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨51298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events290.exact74478RawTerms
def rightRaw : List Term := Proof.Events291.exact74501RawTerms
def group : MergeGroup := .operator 74478 74501
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 74478) (leftOrdinal := 0)
    (rightResult := 74501) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51298⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge74505

namespace LeftMerge74522
def owner : Owner := ⟨.program ⟨257⟩, ⟨51895⟩⟩
def mergeEvent : Nat := 74522
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩] } }
def rhsRaw : List Term := Proof.Events291.exact74519RawTerms
def group : MergeGroup := .relation 74521
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 74521) (rhsResult := 74519)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51892⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 74520 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51892⟩⟩]⟩) (none) 74519) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge74522

namespace LeftMerge74523
def owner : Owner := ⟨.program ⟨257⟩, ⟨51895⟩⟩
def mergeEvent : Nat := 74523
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53162⟩⟩] } }
def rhsRaw : List Term := Proof.Events291.exact74519RawTerms
def group : MergeGroup := .relation 74521
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 74521) (rhsResult := 74519)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51892⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 74520 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51892⟩⟩]⟩) (none) 74519) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53162⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53162⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge74523

namespace LeftMerge74524
def owner : Owner := ⟨.program ⟨257⟩, ⟨51895⟩⟩
def mergeEvent : Nat := 74524
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52223⟩⟩] } }
def rhsRaw : List Term := Proof.Events291.exact74519RawTerms
def group : MergeGroup := .relation 74521
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 74521) (rhsResult := 74519)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51892⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 74520 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51892⟩⟩]⟩) (none) 74519) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52223⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨52223⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge74524

namespace LeftMerge74525
def owner : Owner := ⟨.program ⟨257⟩, ⟨51895⟩⟩
def mergeEvent : Nat := 74525
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events291.exact74519RawTerms
def group : MergeGroup := .relation 74521
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 74521) (rhsResult := 74519)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51892⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 74520 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨51892⟩⟩]⟩) (none) 74519) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨51298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge74525

namespace LeftMerge74530
def owner : Owner := ⟨.program ⟨257⟩, ⟨53165⟩⟩
def mergeEvent : Nat := 74530
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53162⟩⟩] } }
def leftRaw : List Term := Proof.Events291.exact74526RawTerms
def rightRaw : List Term := Proof.Events290.exact74348RawTerms
def group : MergeGroup := .operator 74526 74348
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 74526) (leftOrdinal := 0)
    (rightResult := 74348) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53162⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53162⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7183⟩⟩, ⟨.program ⟨257⟩, ⟨53162⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge74530

namespace LeftMerge74531
def owner : Owner := ⟨.program ⟨257⟩, ⟨53165⟩⟩
def mergeEvent : Nat := 74531
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52223⟩⟩] } }
def leftRaw : List Term := Proof.Events291.exact74526RawTerms
def rightRaw : List Term := Proof.Events290.exact74348RawTerms
def group : MergeGroup := .operator 74526 74348
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 74526) (leftOrdinal := 2)
    (rightResult := 74348) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52223⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨52223⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨50944⟩⟩], [⟨.program ⟨257⟩, ⟨52223⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge74531

namespace LeftMerge74539
def owner : Owner := ⟨.program ⟨257⟩, ⟨53166⟩⟩
def mergeEvent : Nat := 74539
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩] } }
def leftRaw : List Term := Proof.Events291.exact74533RawTerms
def rightRaw : List Term := Proof.Events061.exact15802RawTerms
def group : MergeGroup := .operator 74533 15802
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 74533) (leftOrdinal := 0)
    (rightResult := 15802) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7205⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7131⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7205⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge74539

namespace LeftMerge74540
def owner : Owner := ⟨.program ⟨257⟩, ⟨53166⟩⟩
def mergeEvent : Nat := 74540
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩] } }
def leftRaw : List Term := Proof.Events291.exact74533RawTerms
def rightRaw : List Term := Proof.Events061.exact15802RawTerms
def group : MergeGroup := .operator 74533 15802
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 74533) (leftOrdinal := 1)
    (rightResult := 15802) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7131⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨51298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7131⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge74540

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
