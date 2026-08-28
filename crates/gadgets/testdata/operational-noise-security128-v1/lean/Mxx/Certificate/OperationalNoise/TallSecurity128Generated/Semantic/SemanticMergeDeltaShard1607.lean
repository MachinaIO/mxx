import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge261539
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261539
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42934⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 27)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨42934⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨42934⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261539

namespace LeftMerge261541
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261541
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨42934⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1020.exact261349RawTerms
def group : MergeGroup := .relation 261540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261540) (rhsResult := 261349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨42934⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71082⟩⟩) ⟨68800⟩ 261349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨42934⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261541

namespace LeftMerge261542
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261542
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 26)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨40254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨40254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261542

namespace LeftMerge261544
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261544
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨40254⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1020.exact261349RawTerms
def group : MergeGroup := .relation 261543
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261543) (rhsResult := 261349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨40254⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71082⟩⟩) ⟨68800⟩ 261349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨40254⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261544

namespace LeftMerge261545
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261545
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37578⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 25)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨37578⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨37578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261545

namespace LeftMerge261547
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261547
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨37578⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1020.exact261349RawTerms
def group : MergeGroup := .relation 261546
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261546) (rhsResult := 261349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨37578⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71082⟩⟩) ⟨68800⟩ 261349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨37578⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261547

namespace LeftMerge261548
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261548
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34898⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 24)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34898⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨34898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261548

namespace LeftMerge261550
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261550
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨34898⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1020.exact261349RawTerms
def group : MergeGroup := .relation 261549
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261549) (rhsResult := 261349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨34898⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71082⟩⟩) ⟨68800⟩ 261349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨34898⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261550

namespace LeftMerge261551
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261551
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 22)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261551

namespace LeftMerge261553
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261553
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29234⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1020.exact261349RawTerms
def group : MergeGroup := .relation 261552
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261552) (rhsResult := 261349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨29234⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71082⟩⟩) ⟨68800⟩ 261349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨29234⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261553

namespace LeftMerge261554
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261554
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 21)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26554⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261554

namespace LeftMerge261556
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261556
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26554⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1020.exact261349RawTerms
def group : MergeGroup := .relation 261555
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261555) (rhsResult := 261349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨26554⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71082⟩⟩) ⟨68800⟩ 261349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨26554⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261556

namespace LeftMerge261557
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261557
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 35)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨66251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨66251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261557

namespace LeftMerge261559
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261559
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨66251⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1020.exact261349RawTerms
def group : MergeGroup := .relation 261558
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261558) (rhsResult := 261349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨66251⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71082⟩⟩) ⟨68800⟩ 261349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨66251⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261559

namespace LeftMerge261560
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261560
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩] } }
def leftRaw : List Term := Proof.Events1021.exact261511RawTerms
def rightRaw : List Term := Proof.Events1020.exact261352RawTerms
def group : MergeGroup := .operator 261511 261352
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 261511) (leftOrdinal := 34)
    (rightResult := 261352) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨62986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨71082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261560

namespace LeftMerge261562
def owner : Owner := ⟨.program ⟨257⟩, ⟨71083⟩⟩
def mergeEvent : Nat := 261562
def frameStart : Nat := 260836
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨62986⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }
def rhsRaw : List Term := Proof.Events1020.exact261349RawTerms
def group : MergeGroup := .relation 261561
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 261561) (rhsResult := 261349)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨71082⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨71082⟩⟩) ⟨68800⟩ 261349) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨68800⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨62986⟩⟩], [⟨.program ⟨257⟩, ⟨68800⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge261562

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
