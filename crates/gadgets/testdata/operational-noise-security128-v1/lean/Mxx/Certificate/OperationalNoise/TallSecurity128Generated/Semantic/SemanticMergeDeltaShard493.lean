import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge83408
def owner : Owner := ⟨.program ⟨257⟩, ⟨23507⟩⟩
def mergeEvent : Nat := 83408
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22965⟩⟩] } }
def leftRaw : List Term := Proof.Events325.exact83404RawTerms
def rightRaw : List Term := Proof.Events325.exact83218RawTerms
def group : MergeGroup := .operator 83404 83218
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83404) (leftOrdinal := 2)
    (rightResult := 83218) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22965⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨22965⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21191⟩⟩, ⟨.program ⟨257⟩, ⟨21638⟩⟩], [⟨.program ⟨257⟩, ⟨22965⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83408

namespace LeftMerge83409
def owner : Owner := ⟨.program ⟨257⟩, ⟨23507⟩⟩
def mergeEvent : Nat := 83409
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩] } }
def leftRaw : List Term := Proof.Events325.exact83404RawTerms
def rightRaw : List Term := Proof.Events325.exact83218RawTerms
def group : MergeGroup := .operator 83404 83218
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83404) (leftOrdinal := 1)
    (rightResult := 83218) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7286⟩⟩, ⟨.program ⟨257⟩, ⟨9574⟩⟩, ⟨.program ⟨257⟩, ⟨23505⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83409

namespace LeftMerge83417
def owner : Owner := ⟨.program ⟨257⟩, ⟨24060⟩⟩
def mergeEvent : Nat := 83417
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩] } }
def leftRaw : List Term := Proof.Events325.exact83411RawTerms
def rightRaw : List Term := Proof.Events324.exact83134RawTerms
def group : MergeGroup := .operator 83411 83134
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83411) (leftOrdinal := 0)
    (rightResult := 83134) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨24058⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83417

namespace LeftMerge83418
def owner : Owner := ⟨.program ⟨257⟩, ⟨24060⟩⟩
def mergeEvent : Nat := 83418
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩] } }
def leftRaw : List Term := Proof.Events325.exact83411RawTerms
def rightRaw : List Term := Proof.Events324.exact83134RawTerms
def group : MergeGroup := .operator 83411 83134
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83411) (leftOrdinal := 1)
    (rightResult := 83134) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨24058⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83418

namespace LeftMerge83420
def owner : Owner := ⟨.program ⟨257⟩, ⟨24060⟩⟩
def mergeEvent : Nat := 83420
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23135⟩⟩] } }
def rhsRaw : List Term := Proof.Events324.exact83131RawTerms
def group : MergeGroup := .relation 83419
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83419) (rhsResult := 83131)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨24058⟩⟩) ⟨23135⟩ 83131) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23135⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨23135⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83420

namespace LeftMerge83434
def owner : Owner := ⟨.program ⟨257⟩, ⟨22799⟩⟩
def mergeEvent : Nat := 83434
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22796⟩⟩] } }
def leftRaw : List Term := Proof.Events296.exact75995RawTerms
def rightRaw : List Term := Proof.Events325.exact83428RawTerms
def group : MergeGroup := .operator 75995 83428
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 75995) (leftOrdinal := 0)
    (rightResult := 83428) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨22796⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22796⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83434

namespace LeftMerge83555
def owner : Owner := ⟨.program ⟨257⟩, ⟨23312⟩⟩
def mergeEvent : Nat := 83555
def frameStart : Nat := 83489
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events326.exact83551RawTerms
def rightRaw : List Term := Proof.Events326.exact83549RawTerms
def group : MergeGroup := .operator 83551 83549
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83551) (leftOrdinal := 0)
    (rightResult := 83549) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21856⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83555

namespace LeftMerge83567
def owner : Owner := ⟨.program ⟨257⟩, ⟨24059⟩⟩
def mergeEvent : Nat := 83567
def frameStart : Nat := 83489
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩] } }
def leftRaw : List Term := Proof.Events326.exact83563RawTerms
def rightRaw : List Term := Proof.Events326.exact83540RawTerms
def group : MergeGroup := .operator 83563 83540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83563) (leftOrdinal := 0)
    (rightResult := 83540) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨24058⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83567

namespace LeftMerge83568
def owner : Owner := ⟨.program ⟨257⟩, ⟨24059⟩⟩
def mergeEvent : Nat := 83568
def frameStart : Nat := 83489
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩] } }
def leftRaw : List Term := Proof.Events326.exact83563RawTerms
def rightRaw : List Term := Proof.Events326.exact83540RawTerms
def group : MergeGroup := .operator 83563 83540
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83563) (leftOrdinal := 1)
    (rightResult := 83540) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨24058⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83568

namespace LeftMerge83570
def owner : Owner := ⟨.program ⟨257⟩, ⟨24059⟩⟩
def mergeEvent : Nat := 83570
def frameStart : Nat := 83489
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨21856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23135⟩⟩] } }
def rhsRaw : List Term := Proof.Events326.exact83537RawTerms
def group : MergeGroup := .relation 83569
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83569) (rhsResult := 83537)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨24058⟩⟩) ⟨23135⟩ 83537) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨23135⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨23135⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83570

namespace LeftMerge83578
def owner : Owner := ⟨.program ⟨257⟩, ⟨22202⟩⟩
def mergeEvent : Nat := 83578
def frameStart : Nat := 83489
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨22200⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events326.exact83551RawTerms
def rightRaw : List Term := Proof.Events326.exact83574RawTerms
def group : MergeGroup := .operator 83551 83574
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83551) (leftOrdinal := 0)
    (rightResult := 83574) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22200⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83578

namespace LeftMerge83595
def owner : Owner := ⟨.program ⟨257⟩, ⟨22799⟩⟩
def mergeEvent : Nat := 83595
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }
def rhsRaw : List Term := Proof.Events326.exact83592RawTerms
def group : MergeGroup := .relation 83594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83594) (rhsResult := 83592)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22796⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 83593 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22796⟩⟩]⟩) (none) 83592) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7202⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7202⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83595

namespace LeftMerge83596
def owner : Owner := ⟨.program ⟨257⟩, ⟨22799⟩⟩
def mergeEvent : Nat := 83596
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩] } }
def rhsRaw : List Term := Proof.Events326.exact83592RawTerms
def group : MergeGroup := .relation 83594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83594) (rhsResult := 83592)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22796⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 83593 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22796⟩⟩]⟩) (none) 83592) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83596

namespace LeftMerge83597
def owner : Owner := ⟨.program ⟨257⟩, ⟨22799⟩⟩
def mergeEvent : Nat := 83597
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23135⟩⟩] } }
def rhsRaw : List Term := Proof.Events326.exact83592RawTerms
def group : MergeGroup := .relation 83594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83594) (rhsResult := 83592)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22796⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 83593 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22796⟩⟩]⟩) (none) 83592) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21856⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨23135⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨21856⟩⟩], [⟨.program ⟨257⟩, ⟨23135⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83597

namespace LeftMerge83598
def owner : Owner := ⟨.program ⟨257⟩, ⟨22799⟩⟩
def mergeEvent : Nat := 83598
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events326.exact83592RawTerms
def group : MergeGroup := .relation 83594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 83594) (rhsResult := 83592)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22796⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 83593 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨22796⟩⟩]⟩) (none) 83592) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨22200⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩, ⟨.program ⟨257⟩, ⟨22200⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge83598

namespace LeftMerge83603
def owner : Owner := ⟨.program ⟨257⟩, ⟨24061⟩⟩
def mergeEvent : Nat := 83603
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩] } }
def leftRaw : List Term := Proof.Events326.exact83599RawTerms
def rightRaw : List Term := Proof.Events325.exact83421RawTerms
def group : MergeGroup := .operator 83599 83421
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 83599) (leftOrdinal := 0)
    (rightResult := 83421) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨10694⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨10694⟩⟩], [⟨.program ⟨257⟩, ⟨7181⟩⟩, ⟨.program ⟨257⟩, ⟨24058⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge83603

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
