import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge214577
def owner : Owner := ⟨.program ⟨257⟩, ⟨32699⟩⟩
def mergeEvent : Nat := 214577
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32696⟩⟩] } }
def leftRaw : List Term := Proof.Events811.exact207620RawTerms
def rightRaw : List Term := Proof.Events838.exact214571RawTerms
def group : MergeGroup := .operator 207620 214571
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207620) (leftOrdinal := 0)
    (rightResult := 214571) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨32696⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32696⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214577

namespace LeftMerge214698
def owner : Owner := ⟨.program ⟨257⟩, ⟨33308⟩⟩
def mergeEvent : Nat := 214698
def frameStart : Nat := 214632
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events838.exact214694RawTerms
def rightRaw : List Term := Proof.Events838.exact214692RawTerms
def group : MergeGroup := .operator 214694 214692
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214694) (leftOrdinal := 0)
    (rightResult := 214692) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214698

namespace LeftMerge214710
def owner : Owner := ⟨.program ⟨257⟩, ⟨33893⟩⟩
def mergeEvent : Nat := 214710
def frameStart : Nat := 214632
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩] } }
def leftRaw : List Term := Proof.Events838.exact214706RawTerms
def rightRaw : List Term := Proof.Events838.exact214683RawTerms
def group : MergeGroup := .operator 214706 214683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214706) (leftOrdinal := 0)
    (rightResult := 214683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33892⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214710

namespace LeftMerge214711
def owner : Owner := ⟨.program ⟨257⟩, ⟨33893⟩⟩
def mergeEvent : Nat := 214711
def frameStart : Nat := 214632
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩] } }
def leftRaw : List Term := Proof.Events838.exact214706RawTerms
def rightRaw : List Term := Proof.Events838.exact214683RawTerms
def group : MergeGroup := .operator 214706 214683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214706) (leftOrdinal := 1)
    (rightResult := 214683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33892⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨31828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214711

namespace LeftMerge214713
def owner : Owner := ⟨.program ⟨257⟩, ⟨33893⟩⟩
def mergeEvent : Nat := 214713
def frameStart : Nat := 214632
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33101⟩⟩] } }
def rhsRaw : List Term := Proof.Events838.exact214680RawTerms
def group : MergeGroup := .relation 214712
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 214712) (rhsResult := 214680)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨31828⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨33892⟩⟩) ⟨33101⟩ 214680) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨33101⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨31828⟩⟩], [⟨.program ⟨257⟩, ⟨33101⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214713

namespace LeftMerge214721
def owner : Owner := ⟨.program ⟨257⟩, ⟨32108⟩⟩
def mergeEvent : Nat := 214721
def frameStart : Nat := 214632
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events838.exact214694RawTerms
def rightRaw : List Term := Proof.Events838.exact214717RawTerms
def group : MergeGroup := .operator 214694 214717
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214694) (leftOrdinal := 0)
    (rightResult := 214717) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214721

namespace LeftMerge214738
def owner : Owner := ⟨.program ⟨257⟩, ⟨32699⟩⟩
def mergeEvent : Nat := 214738
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }
def rhsRaw : List Term := Proof.Events838.exact214735RawTerms
def group : MergeGroup := .relation 214737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 214737) (rhsResult := 214735)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 214736 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32696⟩⟩]⟩) (none) 214735) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7204⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7204⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214738

namespace LeftMerge214739
def owner : Owner := ⟨.program ⟨257⟩, ⟨32699⟩⟩
def mergeEvent : Nat := 214739
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩] } }
def rhsRaw : List Term := Proof.Events838.exact214735RawTerms
def group : MergeGroup := .relation 214737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 214737) (rhsResult := 214735)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 214736 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32696⟩⟩]⟩) (none) 214735) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214739

namespace LeftMerge214740
def owner : Owner := ⟨.program ⟨257⟩, ⟨32699⟩⟩
def mergeEvent : Nat := 214740
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33101⟩⟩] } }
def rhsRaw : List Term := Proof.Events838.exact214735RawTerms
def group : MergeGroup := .relation 214737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 214737) (rhsResult := 214735)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 214736 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32696⟩⟩]⟩) (none) 214735) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33101⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨31828⟩⟩], [⟨.program ⟨257⟩, ⟨33101⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214740

namespace LeftMerge214741
def owner : Owner := ⟨.program ⟨257⟩, ⟨32699⟩⟩
def mergeEvent : Nat := 214741
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events838.exact214735RawTerms
def group : MergeGroup := .relation 214737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 214737) (rhsResult := 214735)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32696⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 214736 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨32696⟩⟩]⟩) (none) 214735) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨32106⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨32106⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214741

namespace LeftMerge214746
def owner : Owner := ⟨.program ⟨257⟩, ⟨33895⟩⟩
def mergeEvent : Nat := 214746
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩] } }
def leftRaw : List Term := Proof.Events838.exact214742RawTerms
def rightRaw : List Term := Proof.Events838.exact214564RawTerms
def group : MergeGroup := .operator 214742 214564
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214742) (leftOrdinal := 0)
    (rightResult := 214564) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7182⟩⟩, ⟨.program ⟨257⟩, ⟨33892⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214746

namespace LeftMerge214747
def owner : Owner := ⟨.program ⟨257⟩, ⟨33895⟩⟩
def mergeEvent : Nat := 214747
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33101⟩⟩] } }
def leftRaw : List Term := Proof.Events838.exact214742RawTerms
def rightRaw : List Term := Proof.Events838.exact214564RawTerms
def group : MergeGroup := .operator 214742 214564
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214742) (leftOrdinal := 2)
    (rightResult := 214564) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33101⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨31828⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨33101⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨31828⟩⟩], [⟨.program ⟨257⟩, ⟨33101⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214747

namespace LeftMerge214773
def owner : Owner := ⟨.program ⟨257⟩, ⟨21497⟩⟩
def mergeEvent : Nat := 214773
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21494⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events039.exact10163RawTerms
def rightRaw : List Term := Proof.Events810.exact207528RawTerms
def group : MergeGroup := .operator 10163 207528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 10163) (leftOrdinal := 0)
    (rightResult := 207528) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21494⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21494⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214773

namespace LeftMerge214778
def owner : Owner := ⟨.program ⟨257⟩, ⟨8612⟩⟩
def mergeEvent : Nat := 214778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }
def leftRaw : List Term := Proof.Events810.exact207398RawTerms
def rightRaw : List Term := Proof.Events096.exact24595RawTerms
def group : MergeGroup := .operator 207398 24595
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 207398) (leftOrdinal := 0)
    (rightResult := 24595) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214778

namespace LeftMerge214795
def owner : Owner := ⟨.program ⟨257⟩, ⟨21500⟩⟩
def mergeEvent : Nat := 214795
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21101⟩⟩, ⟨.program ⟨257⟩, ⟨21494⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events839.exact214789RawTerms
def rightRaw : List Term := Proof.Events039.exact10166RawTerms
def group : MergeGroup := .operator 214789 10166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214789) (leftOrdinal := 1)
    (rightResult := 10166) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21494⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21101⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21101⟩⟩, ⟨.program ⟨257⟩, ⟨21494⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge214795

namespace LeftMerge214796
def owner : Owner := ⟨.program ⟨257⟩, ⟨21500⟩⟩
def mergeEvent : Nat := 214796
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21101⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }
def leftRaw : List Term := Proof.Events839.exact214789RawTerms
def rightRaw : List Term := Proof.Events039.exact10166RawTerms
def group : MergeGroup := .operator 214789 10166
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 214789) (leftOrdinal := 0)
    (rightResult := 10166) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7306⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨21101⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨21101⟩⟩], [⟨.program ⟨257⟩, ⟨7306⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge214796

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
