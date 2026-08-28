import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge27520
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27520
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42885⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 27)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42885⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27520

namespace LeftMerge27522
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27522
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42885⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27346RawTerms
def group : MergeGroup := .relation 27521
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27521) (rhsResult := 27346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨42885⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 27346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨42885⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27522

namespace LeftMerge27523
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27523
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 15)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7228⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7228⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27523

namespace LeftMerge27524
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27524
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40205⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 26)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40205⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40205⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27524

namespace LeftMerge27526
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27526
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40205⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27346RawTerms
def group : MergeGroup := .relation 27525
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27525) (rhsResult := 27346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40205⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 27346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40205⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27526

namespace LeftMerge27527
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27527
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 14)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7226⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7226⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27527

namespace LeftMerge27528
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27528
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37529⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 25)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37529⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37529⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27528

namespace LeftMerge27530
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27530
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37529⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27346RawTerms
def group : MergeGroup := .relation 27529
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27529) (rhsResult := 27346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37529⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 27346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37529⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27530

namespace LeftMerge27531
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27531
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 13)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7224⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7224⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27531

namespace LeftMerge27532
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27532
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34849⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 24)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34849⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34849⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27532

namespace LeftMerge27534
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27534
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34849⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27346RawTerms
def group : MergeGroup := .relation 27533
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27533) (rhsResult := 27346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34849⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 27346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34849⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27534

namespace LeftMerge27535
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27535
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 12)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7222⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7222⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27535

namespace LeftMerge27536
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27536
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29185⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 22)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29185⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29185⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27536

namespace LeftMerge27538
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27538
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29185⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27346RawTerms
def group : MergeGroup := .relation 27537
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27537) (rhsResult := 27346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29185⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 27346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29185⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27538

namespace LeftMerge27539
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27539
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 11)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27539

namespace LeftMerge27540
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27540
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26505⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 21)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26505⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26505⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27540

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
