import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge269632
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def mergeEvent : Nat := 269632
def frameStart : Nat := 269549
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events1053.exact269628RawTerms
def rightRaw : List Term := Proof.Events1053.exact269625RawTerms
def group : MergeGroup := .operator 269628 269625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269628) (leftOrdinal := 0)
    (rightResult := 269625) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269632

namespace LeftMerge269641
def owner : Owner := ⟨.program ⟨257⟩, ⟨27831⟩⟩
def mergeEvent : Nat := 269641
def frameStart : Nat := 269549
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27828⟩⟩] } }
def leftRaw : List Term := Proof.Events1053.exact269637RawTerms
def rightRaw : List Term := Proof.Events1053.exact269594RawTerms
def group : MergeGroup := .operator 269637 269594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269637) (leftOrdinal := 0)
    (rightResult := 269594) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27828⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27828⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269641

namespace LeftMerge269642
def owner : Owner := ⟨.program ⟨257⟩, ⟨27831⟩⟩
def mergeEvent : Nat := 269642
def frameStart : Nat := 269549
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12856⟩⟩, ⟨.program ⟨257⟩, ⟨25894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27828⟩⟩] } }
def leftRaw : List Term := Proof.Events1053.exact269637RawTerms
def rightRaw : List Term := Proof.Events1053.exact269594RawTerms
def group : MergeGroup := .operator 269637 269594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269637) (leftOrdinal := 1)
    (rightResult := 269594) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12856⟩⟩, ⟨.program ⟨257⟩, ⟨25894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27828⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨12856⟩⟩, ⟨.program ⟨257⟩, ⟨25894⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27828⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge269642

namespace LeftMerge269644
def owner : Owner := ⟨.program ⟨257⟩, ⟨27831⟩⟩
def mergeEvent : Nat := 269644
def frameStart : Nat := 269549
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨12856⟩⟩, ⟨.program ⟨257⟩, ⟨25894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27359⟩⟩] } }
def rhsRaw : List Term := Proof.Events1053.exact269591RawTerms
def group : MergeGroup := .relation 269643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 269643) (rhsResult := 269591)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨12856⟩⟩, ⟨.program ⟨257⟩, ⟨25894⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨27828⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨27828⟩⟩) ⟨27359⟩ 269591) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27359⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨12856⟩⟩, ⟨.program ⟨257⟩, ⟨25894⟩⟩], [⟨.program ⟨257⟩, ⟨27359⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge269644

namespace LeftMerge269652
def owner : Owner := ⟨.program ⟨257⟩, ⟨26344⟩⟩
def mergeEvent : Nat := 269652
def frameStart : Nat := 269549
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26342⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1053.exact269605RawTerms
def rightRaw : List Term := Proof.Events1053.exact269648RawTerms
def group : MergeGroup := .operator 269605 269648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269605) (leftOrdinal := 0)
    (rightResult := 269648) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26342⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26342⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269652

namespace LeftMerge269669
def owner : Owner := ⟨.program ⟨257⟩, ⟨26769⟩⟩
def mergeEvent : Nat := 269669
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }
def rhsRaw : List Term := Proof.Events1053.exact269666RawTerms
def group : MergeGroup := .relation 269668
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 269668) (rhsResult := 269666)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26766⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 269667 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26766⟩⟩]⟩) (none) 269666) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269669

namespace LeftMerge269670
def owner : Owner := ⟨.program ⟨257⟩, ⟨26769⟩⟩
def mergeEvent : Nat := 269670
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27828⟩⟩] } }
def rhsRaw : List Term := Proof.Events1053.exact269666RawTerms
def group : MergeGroup := .relation 269668
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 269668) (rhsResult := 269666)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26766⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 269667 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26766⟩⟩]⟩) (none) 269666) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27828⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27828⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge269670

namespace LeftMerge269671
def owner : Owner := ⟨.program ⟨257⟩, ⟨26769⟩⟩
def mergeEvent : Nat := 269671
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12856⟩⟩, ⟨.program ⟨257⟩, ⟨25894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27359⟩⟩] } }
def rhsRaw : List Term := Proof.Events1053.exact269666RawTerms
def group : MergeGroup := .relation 269668
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 269668) (rhsResult := 269666)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26766⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 269667 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26766⟩⟩]⟩) (none) 269666) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨12856⟩⟩, ⟨.program ⟨257⟩, ⟨25894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27359⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12856⟩⟩, ⟨.program ⟨257⟩, ⟨25894⟩⟩], [⟨.program ⟨257⟩, ⟨27359⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269671

namespace LeftMerge269672
def owner : Owner := ⟨.program ⟨257⟩, ⟨26769⟩⟩
def mergeEvent : Nat := 269672
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26342⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events1053.exact269666RawTerms
def group : MergeGroup := .relation 269668
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 269668) (rhsResult := 269666)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26766⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 269667 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26766⟩⟩]⟩) (none) 269666) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26342⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26342⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge269672

namespace LeftMerge269677
def owner : Owner := ⟨.program ⟨257⟩, ⟨27830⟩⟩
def mergeEvent : Nat := 269677
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12856⟩⟩, ⟨.program ⟨257⟩, ⟨25894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27359⟩⟩] } }
def leftRaw : List Term := Proof.Events1053.exact269673RawTerms
def rightRaw : List Term := Proof.Events1052.exact269487RawTerms
def group : MergeGroup := .operator 269673 269487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269673) (leftOrdinal := 2)
    (rightResult := 269487) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12856⟩⟩, ⟨.program ⟨257⟩, ⟨25894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27359⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12856⟩⟩, ⟨.program ⟨257⟩, ⟨25894⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27359⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨12856⟩⟩, ⟨.program ⟨257⟩, ⟨25894⟩⟩], [⟨.program ⟨257⟩, ⟨27359⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge269677

namespace LeftMerge269678
def owner : Owner := ⟨.program ⟨257⟩, ⟨27830⟩⟩
def mergeEvent : Nat := 269678
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27828⟩⟩] } }
def leftRaw : List Term := Proof.Events1053.exact269673RawTerms
def rightRaw : List Term := Proof.Events1052.exact269487RawTerms
def group : MergeGroup := .operator 269673 269487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269673) (leftOrdinal := 1)
    (rightResult := 269487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27828⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27828⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨27828⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269678

namespace LeftMerge269686
def owner : Owner := ⟨.program ⟨257⟩, ⟨28084⟩⟩
def mergeEvent : Nat := 269686
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28082⟩⟩] } }
def leftRaw : List Term := Proof.Events1053.exact269680RawTerms
def rightRaw : List Term := Proof.Events1052.exact269403RawTerms
def group : MergeGroup := .operator 269680 269403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269680) (leftOrdinal := 0)
    (rightResult := 269403) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28082⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269686

namespace LeftMerge269687
def owner : Owner := ⟨.program ⟨257⟩, ⟨28084⟩⟩
def mergeEvent : Nat := 269687
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26342⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28082⟩⟩] } }
def leftRaw : List Term := Proof.Events1053.exact269680RawTerms
def rightRaw : List Term := Proof.Events1052.exact269403RawTerms
def group : MergeGroup := .operator 269680 269403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269680) (leftOrdinal := 1)
    (rightResult := 269403) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26342⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28082⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26342⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28082⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge269687

namespace LeftMerge269689
def owner : Owner := ⟨.program ⟨257⟩, ⟨28084⟩⟩
def mergeEvent : Nat := 269689
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26342⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27486⟩⟩] } }
def rhsRaw : List Term := Proof.Events1052.exact269400RawTerms
def group : MergeGroup := .relation 269688
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 269688) (rhsResult := 269400)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26342⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28082⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28082⟩⟩) ⟨27486⟩ 269400) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27486⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩, ⟨.program ⟨257⟩, ⟨26342⟩⟩], [⟨.program ⟨257⟩, ⟨27486⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge269689

namespace LeftMerge269703
def owner : Owner := ⟨.program ⟨257⟩, ⟨26993⟩⟩
def mergeEvent : Nat := 269703
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26990⟩⟩] } }
def leftRaw : List Term := Proof.Events1039.exact266120RawTerms
def rightRaw : List Term := Proof.Events1053.exact269697RawTerms
def group : MergeGroup := .operator 266120 269697
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 266120) (leftOrdinal := 0)
    (rightResult := 269697) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨2883⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨26990⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨2883⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26990⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269703

namespace LeftMerge269824
def owner : Owner := ⟨.program ⟨257⟩, ⟨27736⟩⟩
def mergeEvent : Nat := 269824
def frameStart : Nat := 269758
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26342⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events1053.exact269820RawTerms
def rightRaw : List Term := Proof.Events1053.exact269818RawTerms
def group : MergeGroup := .operator 269820 269818
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 269820) (leftOrdinal := 0)
    (rightResult := 269818) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26342⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26342⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge269824

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
