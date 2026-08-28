import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge193483
def owner : Owner := ⟨.program ⟨257⟩, ⟨45932⟩⟩
def mergeEvent : Nat := 193483
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45929⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192995RawTerms
def rightRaw : List Term := Proof.Events755.exact193477RawTerms
def group : MergeGroup := .operator 192995 193477
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192995) (leftOrdinal := 0)
    (rightResult := 193477) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨45929⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193483

namespace LeftMerge193562
def owner : Owner := ⟨.program ⟨257⟩, ⟨45203⟩⟩
def mergeEvent : Nat := 193562
def frameStart : Nat := 193532
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events756.exact193558RawTerms
def rightRaw : List Term := Proof.Events756.exact193555RawTerms
def group : MergeGroup := .operator 193558 193555
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193558) (leftOrdinal := 0)
    (rightResult := 193555) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14811⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45202⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193562

namespace LeftMerge193592
def owner : Owner := ⟨.program ⟨257⟩, ⟨46756⟩⟩
def mergeEvent : Nat := 193592
def frameStart : Nat := 193532
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events756.exact193588RawTerms
def rightRaw : List Term := Proof.Events756.exact193586RawTerms
def group : MergeGroup := .operator 193588 193586
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193588) (leftOrdinal := 0)
    (rightResult := 193586) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193592

namespace LeftMerge193615
def owner : Owner := ⟨.program ⟨257⟩, ⟨9564⟩⟩
def mergeEvent : Nat := 193615
def frameStart : Nat := 193532
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events756.exact193611RawTerms
def rightRaw : List Term := Proof.Events756.exact193608RawTerms
def group : MergeGroup := .operator 193611 193608
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193611) (leftOrdinal := 0)
    (rightResult := 193608) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193615

namespace LeftMerge193624
def owner : Owner := ⟨.program ⟨257⟩, ⟨47004⟩⟩
def mergeEvent : Nat := 193624
def frameStart : Nat := 193532
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩] } }
def leftRaw : List Term := Proof.Events756.exact193620RawTerms
def rightRaw : List Term := Proof.Events756.exact193577RawTerms
def group : MergeGroup := .operator 193620 193577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193620) (leftOrdinal := 0)
    (rightResult := 193577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47001⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193624

namespace LeftMerge193625
def owner : Owner := ⟨.program ⟨257⟩, ⟨47004⟩⟩
def mergeEvent : Nat := 193625
def frameStart : Nat := 193532
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩] } }
def leftRaw : List Term := Proof.Events756.exact193620RawTerms
def rightRaw : List Term := Proof.Events756.exact193577RawTerms
def group : MergeGroup := .operator 193620 193577
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193620) (leftOrdinal := 1)
    (rightResult := 193577) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47001⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge193625

namespace LeftMerge193627
def owner : Owner := ⟨.program ⟨257⟩, ⟨47004⟩⟩
def mergeEvent : Nat := 193627
def frameStart : Nat := 193532
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46481⟩⟩] } }
def rhsRaw : List Term := Proof.Events756.exact193574RawTerms
def group : MergeGroup := .relation 193626
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 193626) (rhsResult := 193574)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨47001⟩⟩) ⟨46481⟩ 193574) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨46481⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨46481⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge193627

namespace LeftMerge193635
def owner : Owner := ⟨.program ⟨257⟩, ⟨45486⟩⟩
def mergeEvent : Nat := 193635
def frameStart : Nat := 193532
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨45484⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events756.exact193588RawTerms
def rightRaw : List Term := Proof.Events756.exact193631RawTerms
def group : MergeGroup := .operator 193588 193631
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193588) (leftOrdinal := 0)
    (rightResult := 193631) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45484⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨45484⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193635

namespace LeftMerge193652
def owner : Owner := ⟨.program ⟨257⟩, ⟨45932⟩⟩
def mergeEvent : Nat := 193652
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }
def rhsRaw : List Term := Proof.Events756.exact193649RawTerms
def group : MergeGroup := .relation 193651
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 193651) (rhsResult := 193649)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 193650 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩) (none) 193649) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193652

namespace LeftMerge193653
def owner : Owner := ⟨.program ⟨257⟩, ⟨45932⟩⟩
def mergeEvent : Nat := 193653
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩] } }
def rhsRaw : List Term := Proof.Events756.exact193649RawTerms
def group : MergeGroup := .relation 193651
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 193651) (rhsResult := 193649)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 193650 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩) (none) 193649) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge193653

namespace LeftMerge193654
def owner : Owner := ⟨.program ⟨257⟩, ⟨45932⟩⟩
def mergeEvent : Nat := 193654
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46481⟩⟩] } }
def rhsRaw : List Term := Proof.Events756.exact193649RawTerms
def group : MergeGroup := .relation 193651
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 193651) (rhsResult := 193649)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 193650 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩) (none) 193649) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46481⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨46481⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193654

namespace LeftMerge193655
def owner : Owner := ⟨.program ⟨257⟩, ⟨45932⟩⟩
def mergeEvent : Nat := 193655
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45484⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events756.exact193649RawTerms
def group : MergeGroup := .relation 193651
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 193651) (rhsResult := 193649)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 193650 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨45929⟩⟩]⟩) (none) 193649) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45484⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45484⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge193655

namespace LeftMerge193660
def owner : Owner := ⟨.program ⟨257⟩, ⟨47003⟩⟩
def mergeEvent : Nat := 193660
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46481⟩⟩] } }
def leftRaw : List Term := Proof.Events756.exact193656RawTerms
def rightRaw : List Term := Proof.Events755.exact193470RawTerms
def group : MergeGroup := .operator 193656 193470
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193656) (leftOrdinal := 2)
    (rightResult := 193470) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46481⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨46481⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨14811⟩⟩, ⟨.program ⟨257⟩, ⟨45202⟩⟩], [⟨.program ⟨257⟩, ⟨46481⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge193660

namespace LeftMerge193661
def owner : Owner := ⟨.program ⟨257⟩, ⟨47003⟩⟩
def mergeEvent : Nat := 193661
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩] } }
def leftRaw : List Term := Proof.Events756.exact193656RawTerms
def rightRaw : List Term := Proof.Events755.exact193470RawTerms
def group : MergeGroup := .operator 193656 193470
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193656) (leftOrdinal := 1)
    (rightResult := 193470) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩, ⟨.program ⟨257⟩, ⟨47001⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193661

namespace LeftMerge193669
def owner : Owner := ⟨.program ⟨257⟩, ⟨47401⟩⟩
def mergeEvent : Nat := 193669
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47399⟩⟩] } }
def leftRaw : List Term := Proof.Events756.exact193663RawTerms
def rightRaw : List Term := Proof.Events755.exact193386RawTerms
def group : MergeGroup := .operator 193663 193386
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193663) (leftOrdinal := 0)
    (rightResult := 193386) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7195⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47399⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7195⟩⟩, ⟨.program ⟨257⟩, ⟨47399⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge193669

namespace LeftMerge193670
def owner : Owner := ⟨.program ⟨257⟩, ⟨47401⟩⟩
def mergeEvent : Nat := 193670
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45484⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47399⟩⟩] } }
def leftRaw : List Term := Proof.Events756.exact193663RawTerms
def rightRaw : List Term := Proof.Events755.exact193386RawTerms
def group : MergeGroup := .operator 193663 193386
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 193663) (leftOrdinal := 1)
    (rightResult := 193386) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45484⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨47399⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨45484⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨47399⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge193670

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
