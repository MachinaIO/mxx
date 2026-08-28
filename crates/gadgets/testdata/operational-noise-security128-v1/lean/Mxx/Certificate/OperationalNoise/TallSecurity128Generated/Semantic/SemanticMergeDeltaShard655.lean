import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge108734
def owner : Owner := ⟨.program ⟨257⟩, ⟨27692⟩⟩
def mergeEvent : Nat := 108734
def frameStart : Nat := 108674
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events424.exact108730RawTerms
def rightRaw : List Term := Proof.Events424.exact108728RawTerms
def group : MergeGroup := .operator 108730 108728
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108730) (leftOrdinal := 0)
    (rightResult := 108728) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108734

namespace LeftMerge108757
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def mergeEvent : Nat := 108757
def frameStart : Nat := 108674
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events424.exact108753RawTerms
def rightRaw : List Term := Proof.Events424.exact108750RawTerms
def group : MergeGroup := .operator 108753 108750
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108753) (leftOrdinal := 0)
    (rightResult := 108750) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108757

namespace LeftMerge108766
def owner : Owner := ⟨.program ⟨257⟩, ⟨27933⟩⟩
def mergeEvent : Nat := 108766
def frameStart : Nat := 108674
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27930⟩⟩] } }
def leftRaw : List Term := Proof.Events424.exact108762RawTerms
def rightRaw : List Term := Proof.Events424.exact108719RawTerms
def group : MergeGroup := .operator 108762 108719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108762) (leftOrdinal := 0)
    (rightResult := 108719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27930⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27930⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108766

namespace LeftMerge108767
def owner : Owner := ⟨.program ⟨257⟩, ⟨27933⟩⟩
def mergeEvent : Nat := 108767
def frameStart : Nat := 108674
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27930⟩⟩] } }
def leftRaw : List Term := Proof.Events424.exact108762RawTerms
def rightRaw : List Term := Proof.Events424.exact108719RawTerms
def group : MergeGroup := .operator 108762 108719
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108762) (leftOrdinal := 1)
    (rightResult := 108719) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27930⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27930⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108767

namespace LeftMerge108769
def owner : Owner := ⟨.program ⟨257⟩, ⟨27933⟩⟩
def mergeEvent : Nat := 108769
def frameStart : Nat := 108674
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27415⟩⟩] } }
def rhsRaw : List Term := Proof.Events424.exact108716RawTerms
def group : MergeGroup := .relation 108768
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108768) (rhsResult := 108716)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27930⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27930⟩⟩) ⟨27415⟩ 108716) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27415⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], [⟨.program ⟨257⟩, ⟨27415⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108769

namespace LeftMerge108777
def owner : Owner := ⟨.program ⟨257⟩, ⟨26418⟩⟩
def mergeEvent : Nat := 108777
def frameStart : Nat := 108674
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events424.exact108730RawTerms
def rightRaw : List Term := Proof.Events424.exact108773RawTerms
def group : MergeGroup := .operator 108730 108773
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108730) (leftOrdinal := 0)
    (rightResult := 108773) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108777

namespace LeftMerge108794
def owner : Owner := ⟨.program ⟨257⟩, ⟨26862⟩⟩
def mergeEvent : Nat := 108794
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }
def rhsRaw : List Term := Proof.Events424.exact108791RawTerms
def group : MergeGroup := .relation 108793
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108793) (rhsResult := 108791)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26859⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108792 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26859⟩⟩]⟩) (none) 108791) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108794

namespace LeftMerge108795
def owner : Owner := ⟨.program ⟨257⟩, ⟨26862⟩⟩
def mergeEvent : Nat := 108795
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27930⟩⟩] } }
def rhsRaw : List Term := Proof.Events424.exact108791RawTerms
def group : MergeGroup := .relation 108793
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108793) (rhsResult := 108791)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26859⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108792 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26859⟩⟩]⟩) (none) 108791) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27930⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27930⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108795

namespace LeftMerge108796
def owner : Owner := ⟨.program ⟨257⟩, ⟨26862⟩⟩
def mergeEvent : Nat := 108796
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27415⟩⟩] } }
def rhsRaw : List Term := Proof.Events424.exact108791RawTerms
def group : MergeGroup := .relation 108793
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108793) (rhsResult := 108791)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26859⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108792 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26859⟩⟩]⟩) (none) 108791) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27415⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], [⟨.program ⟨257⟩, ⟨27415⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108796

namespace LeftMerge108797
def owner : Owner := ⟨.program ⟨257⟩, ⟨26862⟩⟩
def mergeEvent : Nat := 108797
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events424.exact108791RawTerms
def group : MergeGroup := .relation 108793
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108793) (rhsResult := 108791)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26859⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 108792 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26859⟩⟩]⟩) (none) 108791) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108797

namespace LeftMerge108802
def owner : Owner := ⟨.program ⟨257⟩, ⟨27932⟩⟩
def mergeEvent : Nat := 108802
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27415⟩⟩] } }
def leftRaw : List Term := Proof.Events424.exact108798RawTerms
def rightRaw : List Term := Proof.Events424.exact108612RawTerms
def group : MergeGroup := .operator 108798 108612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108798) (leftOrdinal := 2)
    (rightResult := 108612) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27415⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27415⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨12996⟩⟩, ⟨.program ⟨257⟩, ⟨26118⟩⟩], [⟨.program ⟨257⟩, ⟨27415⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108802

namespace LeftMerge108803
def owner : Owner := ⟨.program ⟨257⟩, ⟨27932⟩⟩
def mergeEvent : Nat := 108803
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27930⟩⟩] } }
def leftRaw : List Term := Proof.Events424.exact108798RawTerms
def rightRaw : List Term := Proof.Events424.exact108612RawTerms
def group : MergeGroup := .operator 108798 108612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108798) (leftOrdinal := 1)
    (rightResult := 108612) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27930⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27930⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27930⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108803

namespace LeftMerge108811
def owner : Owner := ⟨.program ⟨257⟩, ⟨28316⟩⟩
def mergeEvent : Nat := 108811
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩] } }
def leftRaw : List Term := Proof.Events425.exact108805RawTerms
def rightRaw : List Term := Proof.Events423.exact108528RawTerms
def group : MergeGroup := .operator 108805 108528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108805) (leftOrdinal := 0)
    (rightResult := 108528) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28314⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108811

namespace LeftMerge108812
def owner : Owner := ⟨.program ⟨257⟩, ⟨28316⟩⟩
def mergeEvent : Nat := 108812
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩] } }
def leftRaw : List Term := Proof.Events425.exact108805RawTerms
def rightRaw : List Term := Proof.Events423.exact108528RawTerms
def group : MergeGroup := .operator 108805 108528
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 108805) (leftOrdinal := 1)
    (rightResult := 108528) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28314⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108812

namespace LeftMerge108814
def owner : Owner := ⟨.program ⟨257⟩, ⟨28316⟩⟩
def mergeEvent : Nat := 108814
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27570⟩⟩] } }
def rhsRaw : List Term := Proof.Events423.exact108525RawTerms
def group : MergeGroup := .relation 108813
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 108813) (rhsResult := 108525)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28314⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28314⟩⟩) ⟨27570⟩ 108525) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27570⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩, ⟨.program ⟨257⟩, ⟨26416⟩⟩], [⟨.program ⟨257⟩, ⟨27570⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge108814

namespace LeftMerge108828
def owner : Owner := ⟨.program ⟨257⟩, ⟨27179⟩⟩
def mergeEvent : Nat := 108828
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩] } }
def leftRaw : List Term := Proof.Events411.exact105245RawTerms
def rightRaw : List Term := Proof.Events425.exact108822RawTerms
def group : MergeGroup := .operator 105245 108822
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 105245) (leftOrdinal := 0)
    (rightResult := 108822) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨9846⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27176⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨9846⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27176⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge108828

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
