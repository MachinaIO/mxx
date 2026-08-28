import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge20682
def owner : Owner := ⟨.program ⟨257⟩, ⟨26765⟩⟩
def mergeEvent : Nat := 20682
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26762⟩⟩] } }
def leftRaw : List Term := Proof.Events067.exact17169RawTerms
def rightRaw : List Term := Proof.Events080.exact20676RawTerms
def group : MergeGroup := .operator 17169 20676
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 17169) (leftOrdinal := 0)
    (rightResult := 20676) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨26762⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26762⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge20682

namespace LeftMerge20761
def owner : Owner := ⟨.program ⟨257⟩, ⟨25887⟩⟩
def mergeEvent : Nat := 20761
def frameStart : Nat := 20731
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events081.exact20757RawTerms
def rightRaw : List Term := Proof.Events081.exact20754RawTerms
def group : MergeGroup := .operator 20757 20754
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 20757) (leftOrdinal := 0)
    (rightResult := 20754) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12851⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨25886⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge20761

namespace LeftMerge20791
def owner : Owner := ⟨.program ⟨257⟩, ⟨27652⟩⟩
def mergeEvent : Nat := 20791
def frameStart : Nat := 20731
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events081.exact20787RawTerms
def rightRaw : List Term := Proof.Events081.exact20785RawTerms
def group : MergeGroup := .operator 20787 20785
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 20787) (leftOrdinal := 0)
    (rightResult := 20785) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge20791

namespace LeftMerge20814
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def mergeEvent : Nat := 20814
def frameStart : Nat := 20731
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events081.exact20810RawTerms
def rightRaw : List Term := Proof.Events081.exact20807RawTerms
def group : MergeGroup := .operator 20810 20807
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 20810) (leftOrdinal := 0)
    (rightResult := 20807) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge20814

namespace LeftMerge20823
def owner : Owner := ⟨.program ⟨257⟩, ⟨27826⟩⟩
def mergeEvent : Nat := 20823
def frameStart : Nat := 20731
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27823⟩⟩] } }
def leftRaw : List Term := Proof.Events081.exact20819RawTerms
def rightRaw : List Term := Proof.Events081.exact20776RawTerms
def group : MergeGroup := .operator 20819 20776
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 20819) (leftOrdinal := 1)
    (rightResult := 20776) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27823⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27823⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge20823

namespace LeftMerge20825
def owner : Owner := ⟨.program ⟨257⟩, ⟨27826⟩⟩
def mergeEvent : Nat := 20825
def frameStart : Nat := 20731
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27357⟩⟩] } }
def rhsRaw : List Term := Proof.Events081.exact20773RawTerms
def group : MergeGroup := .relation 20824
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 20824) (rhsResult := 20773)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27823⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27823⟩⟩) ⟨27357⟩ 20773) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27357⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], [⟨.program ⟨257⟩, ⟨27357⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge20825

namespace LeftMerge20826
def owner : Owner := ⟨.program ⟨257⟩, ⟨27826⟩⟩
def mergeEvent : Nat := 20826
def frameStart : Nat := 20731
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27823⟩⟩] } }
def leftRaw : List Term := Proof.Events081.exact20819RawTerms
def rightRaw : List Term := Proof.Events081.exact20776RawTerms
def group : MergeGroup := .operator 20819 20776
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 20819) (leftOrdinal := 0)
    (rightResult := 20776) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27823⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27823⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge20826

namespace LeftMerge20834
def owner : Owner := ⟨.program ⟨257⟩, ⟨26340⟩⟩
def mergeEvent : Nat := 20834
def frameStart : Nat := 20731
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events081.exact20787RawTerms
def rightRaw : List Term := Proof.Events081.exact20830RawTerms
def group : MergeGroup := .operator 20787 20830
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 20787) (leftOrdinal := 0)
    (rightResult := 20830) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge20834

namespace LeftMerge20851
def owner : Owner := ⟨.program ⟨257⟩, ⟨26765⟩⟩
def mergeEvent : Nat := 20851
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27357⟩⟩] } }
def rhsRaw : List Term := Proof.Events081.exact20848RawTerms
def group : MergeGroup := .relation 20850
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 20850) (rhsResult := 20848)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26762⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 20849 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26762⟩⟩]⟩) (none) 20848) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27357⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], [⟨.program ⟨257⟩, ⟨27357⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge20851

namespace LeftMerge20852
def owner : Owner := ⟨.program ⟨257⟩, ⟨26765⟩⟩
def mergeEvent : Nat := 20852
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27823⟩⟩] } }
def rhsRaw : List Term := Proof.Events081.exact20848RawTerms
def group : MergeGroup := .relation 20850
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 20850) (rhsResult := 20848)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26762⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 20849 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26762⟩⟩]⟩) (none) 20848) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27823⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27823⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge20852

namespace LeftMerge20853
def owner : Owner := ⟨.program ⟨257⟩, ⟨26765⟩⟩
def mergeEvent : Nat := 20853
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events081.exact20848RawTerms
def group : MergeGroup := .relation 20850
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 20850) (rhsResult := 20848)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26762⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 20849 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26762⟩⟩]⟩) (none) 20848) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge20853

namespace LeftMerge20854
def owner : Owner := ⟨.program ⟨257⟩, ⟨26765⟩⟩
def mergeEvent : Nat := 20854
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }
def rhsRaw : List Term := Proof.Events081.exact20848RawTerms
def group : MergeGroup := .relation 20850
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 20850) (rhsResult := 20848)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26762⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 20849 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26762⟩⟩]⟩) (none) 20848) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge20854

namespace LeftMerge20859
def owner : Owner := ⟨.program ⟨257⟩, ⟨27825⟩⟩
def mergeEvent : Nat := 20859
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27357⟩⟩] } }
def leftRaw : List Term := Proof.Events081.exact20855RawTerms
def rightRaw : List Term := Proof.Events080.exact20669RawTerms
def group : MergeGroup := .operator 20855 20669
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 20855) (leftOrdinal := 2)
    (rightResult := 20669) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27357⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27357⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨12851⟩⟩, ⟨.program ⟨257⟩, ⟨25886⟩⟩], [⟨.program ⟨257⟩, ⟨27357⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge20859

namespace LeftMerge20860
def owner : Owner := ⟨.program ⟨257⟩, ⟨27825⟩⟩
def mergeEvent : Nat := 20860
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27823⟩⟩] } }
def leftRaw : List Term := Proof.Events081.exact20855RawTerms
def rightRaw : List Term := Proof.Events080.exact20669RawTerms
def group : MergeGroup := .operator 20855 20669
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 20855) (leftOrdinal := 1)
    (rightResult := 20669) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27823⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27823⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27823⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge20860

namespace LeftMerge20868
def owner : Owner := ⟨.program ⟨257⟩, ⟨28073⟩⟩
def mergeEvent : Nat := 20868
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28071⟩⟩] } }
def leftRaw : List Term := Proof.Events081.exact20862RawTerms
def rightRaw : List Term := Proof.Events080.exact20566RawTerms
def group : MergeGroup := .operator 20862 20566
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 20862) (leftOrdinal := 1)
    (rightResult := 20566) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28071⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28071⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge20868

namespace LeftMerge20870
def owner : Owner := ⟨.program ⟨257⟩, ⟨28073⟩⟩
def mergeEvent : Nat := 20870
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26338⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27483⟩⟩] } }
def rhsRaw : List Term := Proof.Events080.exact20563RawTerms
def group : MergeGroup := .relation 20869
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 20869) (rhsResult := 20563)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26338⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28071⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28071⟩⟩) ⟨27483⟩ 20563) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27483⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2371⟩⟩, ⟨.program ⟨257⟩, ⟨26338⟩⟩], [⟨.program ⟨257⟩, ⟨27483⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge20870

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
