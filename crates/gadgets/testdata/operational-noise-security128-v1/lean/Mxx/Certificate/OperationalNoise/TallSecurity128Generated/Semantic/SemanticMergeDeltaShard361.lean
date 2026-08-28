import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge61363
def owner : Owner := ⟨.program ⟨257⟩, ⟨10791⟩⟩
def mergeEvent : Nat := 61363
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }
def leftRaw : List Term := Proof.Events238.exact61148RawTerms
def rightRaw : List Term := Proof.Events067.exact17158RawTerms
def group : MergeGroup := .operator 61148 17158
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61148) (leftOrdinal := 0)
    (rightResult := 17158) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61363

namespace LeftMerge61376
def owner : Owner := ⟨.program ⟨257⟩, ⟨48662⟩⟩
def mergeEvent : Nat := 61376
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48659⟩⟩] } }
def leftRaw : List Term := Proof.Events239.exact61370RawTerms
def rightRaw : List Term := Proof.Events239.exact61359RawTerms
def group : MergeGroup := .operator 61370 61359
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61370) (leftOrdinal := 0)
    (rightResult := 61359) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨48659⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48659⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61376

namespace LeftMerge61455
def owner : Owner := ⟨.program ⟨257⟩, ⟨48003⟩⟩
def mergeEvent : Nat := 61455
def frameStart : Nat := 61425
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events240.exact61451RawTerms
def rightRaw : List Term := Proof.Events240.exact61448RawTerms
def group : MergeGroup := .operator 61451 61448
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61451) (leftOrdinal := 0)
    (rightResult := 61448) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15186⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48002⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61455

namespace LeftMerge61485
def owner : Owner := ⟨.program ⟨257⟩, ⟨49456⟩⟩
def mergeEvent : Nat := 61485
def frameStart : Nat := 61425
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events240.exact61481RawTerms
def rightRaw : List Term := Proof.Events240.exact61479RawTerms
def group : MergeGroup := .operator 61481 61479
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61481) (leftOrdinal := 0)
    (rightResult := 61479) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61485

namespace LeftMerge61508
def owner : Owner := ⟨.program ⟨257⟩, ⟨9567⟩⟩
def mergeEvent : Nat := 61508
def frameStart : Nat := 61425
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }
def leftRaw : List Term := Proof.Events240.exact61504RawTerms
def rightRaw : List Term := Proof.Events240.exact61501RawTerms
def group : MergeGroup := .operator 61504 61501
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61504) (leftOrdinal := 0)
    (rightResult := 61501) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9565⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61508

namespace LeftMerge61517
def owner : Owner := ⟨.program ⟨257⟩, ⟨49739⟩⟩
def mergeEvent : Nat := 61517
def frameStart : Nat := 61425
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49736⟩⟩] } }
def leftRaw : List Term := Proof.Events240.exact61513RawTerms
def rightRaw : List Term := Proof.Events240.exact61470RawTerms
def group : MergeGroup := .operator 61513 61470
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61513) (leftOrdinal := 0)
    (rightResult := 61470) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49736⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49736⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61517

namespace LeftMerge61518
def owner : Owner := ⟨.program ⟨257⟩, ⟨49739⟩⟩
def mergeEvent : Nat := 61518
def frameStart : Nat := 61425
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49736⟩⟩] } }
def leftRaw : List Term := Proof.Events240.exact61513RawTerms
def rightRaw : List Term := Proof.Events240.exact61470RawTerms
def group : MergeGroup := .operator 61513 61470
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61513) (leftOrdinal := 1)
    (rightResult := 61470) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49736⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49736⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61518

namespace LeftMerge61520
def owner : Owner := ⟨.program ⟨257⟩, ⟨49739⟩⟩
def mergeEvent : Nat := 61520
def frameStart : Nat := 61425
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49191⟩⟩] } }
def rhsRaw : List Term := Proof.Events240.exact61467RawTerms
def group : MergeGroup := .relation 61519
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 61519) (rhsResult := 61467)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨49736⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨49736⟩⟩) ⟨49191⟩ 61467) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], [⟨.program ⟨257⟩, ⟨49191⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61520

namespace LeftMerge61528
def owner : Owner := ⟨.program ⟨257⟩, ⟨48206⟩⟩
def mergeEvent : Nat := 61528
def frameStart : Nat := 61425
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events240.exact61481RawTerms
def rightRaw : List Term := Proof.Events240.exact61524RawTerms
def group : MergeGroup := .operator 61481 61524
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61481) (leftOrdinal := 0)
    (rightResult := 61524) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61528

namespace LeftMerge61545
def owner : Owner := ⟨.program ⟨257⟩, ⟨48662⟩⟩
def mergeEvent : Nat := 61545
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }
def rhsRaw : List Term := Proof.Events240.exact61542RawTerms
def group : MergeGroup := .relation 61544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 61544) (rhsResult := 61542)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48659⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 61543 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48659⟩⟩]⟩) (none) 61542) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61545

namespace LeftMerge61546
def owner : Owner := ⟨.program ⟨257⟩, ⟨48662⟩⟩
def mergeEvent : Nat := 61546
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49736⟩⟩] } }
def rhsRaw : List Term := Proof.Events240.exact61542RawTerms
def group : MergeGroup := .relation 61544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 61544) (rhsResult := 61542)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48659⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 61543 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48659⟩⟩]⟩) (none) 61542) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49736⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49736⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61546

namespace LeftMerge61547
def owner : Owner := ⟨.program ⟨257⟩, ⟨48662⟩⟩
def mergeEvent : Nat := 61547
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49191⟩⟩] } }
def rhsRaw : List Term := Proof.Events240.exact61542RawTerms
def group : MergeGroup := .relation 61544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 61544) (rhsResult := 61542)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48659⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 61543 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48659⟩⟩]⟩) (none) 61542) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49191⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], [⟨.program ⟨257⟩, ⟨49191⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61547

namespace LeftMerge61548
def owner : Owner := ⟨.program ⟨257⟩, ⟨48662⟩⟩
def mergeEvent : Nat := 61548
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events240.exact61542RawTerms
def group : MergeGroup := .relation 61544
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 61544) (rhsResult := 61542)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48659⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 61543 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48659⟩⟩]⟩) (none) 61542) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48204⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨48204⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61548

namespace LeftMerge61553
def owner : Owner := ⟨.program ⟨257⟩, ⟨49738⟩⟩
def mergeEvent : Nat := 61553
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49191⟩⟩] } }
def leftRaw : List Term := Proof.Events240.exact61549RawTerms
def rightRaw : List Term := Proof.Events239.exact61352RawTerms
def group : MergeGroup := .operator 61549 61352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61549) (leftOrdinal := 2)
    (rightResult := 61352) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49191⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49191⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩, ⟨.program ⟨257⟩, ⟨15186⟩⟩, ⟨.program ⟨257⟩, ⟨48002⟩⟩], [⟨.program ⟨257⟩, ⟨49191⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge61553

namespace LeftMerge61554
def owner : Owner := ⟨.program ⟨257⟩, ⟨49738⟩⟩
def mergeEvent : Nat := 61554
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49736⟩⟩] } }
def leftRaw : List Term := Proof.Events240.exact61549RawTerms
def rightRaw : List Term := Proof.Events239.exact61352RawTerms
def group : MergeGroup := .operator 61549 61352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61549) (leftOrdinal := 1)
    (rightResult := 61352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49736⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49736⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7302⟩⟩, ⟨.program ⟨257⟩, ⟨9565⟩⟩, ⟨.program ⟨257⟩, ⟨49736⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61554

namespace LeftMerge61562
def owner : Owner := ⟨.program ⟨257⟩, ⟨50206⟩⟩
def mergeEvent : Nat := 61562
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50204⟩⟩] } }
def leftRaw : List Term := Proof.Events240.exact61556RawTerms
def rightRaw : List Term := Proof.Events239.exact61263RawTerms
def group : MergeGroup := .operator 61556 61263
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 61556) (leftOrdinal := 0)
    (rightResult := 61263) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50204⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11118⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge61562

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
