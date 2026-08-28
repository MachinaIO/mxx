import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge87411
def owner : Owner := ⟨.program ⟨257⟩, ⟨36132⟩⟩
def mergeEvent : Nat := 87411
def frameStart : Nat := 87345
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events341.exact87407RawTerms
def rightRaw : List Term := Proof.Events341.exact87405RawTerms
def group : MergeGroup := .operator 87407 87405
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87407) (leftOrdinal := 0)
    (rightResult := 87405) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34796⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87411

namespace LeftMerge87423
def owner : Owner := ⟨.program ⟨257⟩, ⟨36774⟩⟩
def mergeEvent : Nat := 87423
def frameStart : Nat := 87345
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩] } }
def leftRaw : List Term := Proof.Events341.exact87419RawTerms
def rightRaw : List Term := Proof.Events341.exact87396RawTerms
def group : MergeGroup := .operator 87419 87396
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87419) (leftOrdinal := 0)
    (rightResult := 87396) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36773⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87423

namespace LeftMerge87424
def owner : Owner := ⟨.program ⟨257⟩, ⟨36774⟩⟩
def mergeEvent : Nat := 87424
def frameStart : Nat := 87345
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩] } }
def leftRaw : List Term := Proof.Events341.exact87419RawTerms
def rightRaw : List Term := Proof.Events341.exact87396RawTerms
def group : MergeGroup := .operator 87419 87396
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87419) (leftOrdinal := 1)
    (rightResult := 87396) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨36773⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87424

namespace LeftMerge87426
def owner : Owner := ⟨.program ⟨257⟩, ⟨36774⟩⟩
def mergeEvent : Nat := 87426
def frameStart : Nat := 87345
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35954⟩⟩] } }
def rhsRaw : List Term := Proof.Events341.exact87393RawTerms
def group : MergeGroup := .relation 87425
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87425) (rhsResult := 87393)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨36773⟩⟩) ⟨35954⟩ 87393) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨35954⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35954⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87426

namespace LeftMerge87434
def owner : Owner := ⟨.program ⟨257⟩, ⟨35039⟩⟩
def mergeEvent : Nat := 87434
def frameStart : Nat := 87345
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨35037⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events341.exact87407RawTerms
def rightRaw : List Term := Proof.Events341.exact87430RawTerms
def group : MergeGroup := .operator 87407 87430
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87407) (leftOrdinal := 0)
    (rightResult := 87430) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨35037⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87434

namespace LeftMerge87451
def owner : Owner := ⟨.program ⟨257⟩, ⟨35615⟩⟩
def mergeEvent : Nat := 87451
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩] } }
def rhsRaw : List Term := Proof.Events341.exact87448RawTerms
def group : MergeGroup := .relation 87450
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87450) (rhsResult := 87448)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 87449 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩) (none) 87448) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87451

namespace LeftMerge87452
def owner : Owner := ⟨.program ⟨257⟩, ⟨35615⟩⟩
def mergeEvent : Nat := 87452
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩] } }
def rhsRaw : List Term := Proof.Events341.exact87448RawTerms
def group : MergeGroup := .relation 87450
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87450) (rhsResult := 87448)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 87449 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩) (none) 87448) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87452

namespace LeftMerge87453
def owner : Owner := ⟨.program ⟨257⟩, ⟨35615⟩⟩
def mergeEvent : Nat := 87453
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35954⟩⟩] } }
def rhsRaw : List Term := Proof.Events341.exact87448RawTerms
def group : MergeGroup := .relation 87450
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87450) (rhsResult := 87448)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 87449 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩) (none) 87448) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35954⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35954⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87453

namespace LeftMerge87454
def owner : Owner := ⟨.program ⟨257⟩, ⟨35615⟩⟩
def mergeEvent : Nat := 87454
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events341.exact87448RawTerms
def group : MergeGroup := .relation 87450
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87450) (rhsResult := 87448)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 87449 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35612⟩⟩]⟩) (none) 87448) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨35037⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87454

namespace LeftMerge87459
def owner : Owner := ⟨.program ⟨257⟩, ⟨36776⟩⟩
def mergeEvent : Nat := 87459
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩] } }
def leftRaw : List Term := Proof.Events341.exact87455RawTerms
def rightRaw : List Term := Proof.Events340.exact87277RawTerms
def group : MergeGroup := .operator 87455 87277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87455) (leftOrdinal := 0)
    (rightResult := 87277) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36773⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87459

namespace LeftMerge87460
def owner : Owner := ⟨.program ⟨257⟩, ⟨36776⟩⟩
def mergeEvent : Nat := 87460
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35954⟩⟩] } }
def leftRaw : List Term := Proof.Events341.exact87455RawTerms
def rightRaw : List Term := Proof.Events340.exact87277RawTerms
def group : MergeGroup := .operator 87455 87277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87455) (leftOrdinal := 2)
    (rightResult := 87277) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35954⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35954⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨34796⟩⟩], [⟨.program ⟨257⟩, ⟨35954⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87460

namespace LeftMerge87468
def owner : Owner := ⟨.program ⟨257⟩, ⟨36777⟩⟩
def mergeEvent : Nat := 87468
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩] } }
def leftRaw : List Term := Proof.Events341.exact87462RawTerms
def rightRaw : List Term := Proof.Events061.exact15642RawTerms
def group : MergeGroup := .operator 87462 15642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87462) (leftOrdinal := 0)
    (rightResult := 15642) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7221⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7221⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87468

namespace LeftMerge87469
def owner : Owner := ⟨.program ⟨257⟩, ⟨36777⟩⟩
def mergeEvent : Nat := 87469
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩] } }
def leftRaw : List Term := Proof.Events341.exact87462RawTerms
def rightRaw : List Term := Proof.Events061.exact15642RawTerms
def group : MergeGroup := .operator 87462 15642
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 87462) (leftOrdinal := 1)
    (rightResult := 15642) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7163⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87469

namespace LeftMerge87471
def owner : Owner := ⟨.program ⟨257⟩, ⟨36777⟩⟩
def mergeEvent : Nat := 87471
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events061.exact15635RawTerms
def group : MergeGroup := .relation 87470
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 87470) (rhsResult := 15635)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨7163⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨7163⟩⟩) ⟨7047⟩ 15635) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6842⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6842⟩⟩, ⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨35037⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87471

namespace LeftMerge87485
def owner : Owner := ⟨.program ⟨257⟩, ⟨31115⟩⟩
def mergeEvent : Nat := 87485
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩] } }
def leftRaw : List Term := Proof.Events308.exact79073RawTerms
def rightRaw : List Term := Proof.Events341.exact87479RawTerms
def group : MergeGroup := .operator 79073 87479
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79073) (leftOrdinal := 0)
    (rightResult := 87479) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge87485

namespace LeftMerge87486
def owner : Owner := ⟨.program ⟨257⟩, ⟨31115⟩⟩
def mergeEvent : Nat := 87486
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩] } }
def leftRaw : List Term := Proof.Events308.exact79073RawTerms
def rightRaw : List Term := Proof.Events341.exact87479RawTerms
def group : MergeGroup := .operator 79073 87479
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 79073) (leftOrdinal := 1)
    (rightResult := 87479) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨31113⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨29136⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨31113⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge87486

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
