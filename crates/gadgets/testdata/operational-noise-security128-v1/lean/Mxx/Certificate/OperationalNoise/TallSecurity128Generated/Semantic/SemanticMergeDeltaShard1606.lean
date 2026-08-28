import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge261521
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261521
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 11)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7220⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7220⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261521

namespace LeftMerge261522
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261522
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 10)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7218⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7218⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261522

namespace LeftMerge261523
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261523
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 9)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7216⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7216⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261523

namespace LeftMerge261524
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261524
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 8)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7214⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7214⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261524

namespace LeftMerge261525
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261525
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 7)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7212⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7212⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261525

namespace LeftMerge261526
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261526
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 6)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7210⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7210⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261526

namespace LeftMerge261527
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261527
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 5)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7208⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7208⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261527

namespace LeftMerge261528
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261528
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 4)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7206⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7206⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261528

namespace LeftMerge261529
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261529
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 3)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7204⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261529

namespace LeftMerge261530
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261530
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 2)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7202⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261530

namespace LeftMerge261531
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261531
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 1)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7200⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7200⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261531

namespace LeftMerge261532
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261532
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 0)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7198⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7198⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge261532

namespace LeftMerge261533
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261533
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 29)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261533

namespace LeftMerge261535
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261535
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48298⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1020.exact261349RawTerms
def group : MergeGroup := .relation 261534
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261534) (rhsResult := 261349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48298⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71082⟩⟩) ⟨68800⟩ 261349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48298⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261535

namespace LeftMerge261536
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261536
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 28)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261536

namespace LeftMerge261538
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261538
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45618⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1020.exact261349RawTerms
def group : MergeGroup := .relation 261537
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261537) (rhsResult := 261349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71082⟩⟩) ⟨68800⟩ 261349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨45618⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261538

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
