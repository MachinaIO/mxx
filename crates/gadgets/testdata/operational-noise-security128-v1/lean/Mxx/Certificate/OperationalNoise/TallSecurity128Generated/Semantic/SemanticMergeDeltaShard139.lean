import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge27563
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27563
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 5)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27563

namespace LeftMerge27564
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27564
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50995⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 30)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨50995⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27564

namespace LeftMerge27566
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27566
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨50995⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27346RawTerms
def group : MergeGroup := .relation 27565
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27565) (rhsResult := 27346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 27346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨50995⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27566

namespace LeftMerge27567
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27567
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 4)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27567

namespace LeftMerge27568
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27568
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31940⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 23)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31940⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27568

namespace LeftMerge27570
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27570
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31940⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27346RawTerms
def group : MergeGroup := .relation 27569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27569) (rhsResult := 27346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 27346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨31940⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27570

namespace LeftMerge27571
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27571
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 3)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27571

namespace LeftMerge27572
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27572
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 20)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27572

namespace LeftMerge27574
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27574
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21920⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27346RawTerms
def group : MergeGroup := .relation 27573
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27573) (rhsResult := 27346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 27346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21920⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27574

namespace LeftMerge27575
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27575
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 2)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27575

namespace LeftMerge27576
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27576
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18700⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 19)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨18700⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27576

namespace LeftMerge27578
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27578
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨18700⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27346RawTerms
def group : MergeGroup := .relation 27577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27577) (rhsResult := 27346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 27346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨18700⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27578

namespace LeftMerge27579
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27579
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 1)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27579

namespace LeftMerge27580
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27580
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15895⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 18)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨15895⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27580

namespace LeftMerge27582
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27582
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨15895⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }
def rhsRaw : List Term := Proof.Events106.exact27346RawTerms
def group : MergeGroup := .relation 27581
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 27581) (rhsResult := 27346)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨70968⟩⟩) ⟨68778⟩ 27346) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68778⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨15895⟩⟩], [⟨.program ⟨257⟩, ⟨68778⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge27582

namespace LeftMerge27583
def owner : Owner := ⟨.program ⟨257⟩, ⟨70969⟩⟩
def mergeEvent : Nat := 27583
def frameStart : Nat := 26833
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩] } }
def leftRaw : List Term := Proof.Events107.exact27508RawTerms
def rightRaw : List Term := Proof.Events106.exact27349RawTerms
def group : MergeGroup := .operator 27508 27349
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 27508) (leftOrdinal := 0)
    (rightResult := 27349) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨70968⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨70968⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge27583

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
