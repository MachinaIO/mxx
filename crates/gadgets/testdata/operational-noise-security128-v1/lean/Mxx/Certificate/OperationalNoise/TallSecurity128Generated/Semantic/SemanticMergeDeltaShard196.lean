import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge35632
def owner : Owner := ⟨.program ⟨257⟩, ⟨9546⟩⟩
def mergeEvent : Nat := 35632
def frameStart : Nat := 35549
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }
def leftRaw : List Term := Proof.Events139.exact35628RawTerms
def rightRaw : List Term := Proof.Events139.exact35625RawTerms
def group : MergeGroup := .operator 35628 35625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35628) (leftOrdinal := 0)
    (rightResult := 35625) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9544⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35632

namespace LeftMerge35641
def owner : Owner := ⟨.program ⟨257⟩, ⟨28021⟩⟩
def mergeEvent : Nat := 35641
def frameStart : Nat := 35549
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28018⟩⟩] } }
def leftRaw : List Term := Proof.Events139.exact35637RawTerms
def rightRaw : List Term := Proof.Events139.exact35594RawTerms
def group : MergeGroup := .operator 35637 35594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35637) (leftOrdinal := 0)
    (rightResult := 35594) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28018⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28018⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35641

namespace LeftMerge35642
def owner : Owner := ⟨.program ⟨257⟩, ⟨28021⟩⟩
def mergeEvent : Nat := 35642
def frameStart : Nat := 35549
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13116⟩⟩, ⟨.program ⟨257⟩, ⟨26310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28018⟩⟩] } }
def leftRaw : List Term := Proof.Events139.exact35637RawTerms
def rightRaw : List Term := Proof.Events139.exact35594RawTerms
def group : MergeGroup := .operator 35637 35594
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35637) (leftOrdinal := 1)
    (rightResult := 35594) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13116⟩⟩, ⟨.program ⟨257⟩, ⟨26310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28018⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13116⟩⟩, ⟨.program ⟨257⟩, ⟨26310⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28018⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35642

namespace LeftMerge35644
def owner : Owner := ⟨.program ⟨257⟩, ⟨28021⟩⟩
def mergeEvent : Nat := 35644
def frameStart : Nat := 35549
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13116⟩⟩, ⟨.program ⟨257⟩, ⟨26310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27463⟩⟩] } }
def rhsRaw : List Term := Proof.Events139.exact35591RawTerms
def group : MergeGroup := .relation 35643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35643) (rhsResult := 35591)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13116⟩⟩, ⟨.program ⟨257⟩, ⟨26310⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28018⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28018⟩⟩) ⟨27463⟩ 35591) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27463⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13116⟩⟩, ⟨.program ⟨257⟩, ⟨26310⟩⟩], [⟨.program ⟨257⟩, ⟨27463⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35644

namespace LeftMerge35652
def owner : Owner := ⟨.program ⟨257⟩, ⟨26482⟩⟩
def mergeEvent : Nat := 35652
def frameStart : Nat := 35549
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26480⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events139.exact35605RawTerms
def rightRaw : List Term := Proof.Events139.exact35648RawTerms
def group : MergeGroup := .operator 35605 35648
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35605) (leftOrdinal := 0)
    (rightResult := 35648) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26480⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26480⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35652

namespace LeftMerge35669
def owner : Owner := ⟨.program ⟨257⟩, ⟨26942⟩⟩
def mergeEvent : Nat := 35669
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }
def rhsRaw : List Term := Proof.Events139.exact35666RawTerms
def group : MergeGroup := .relation 35668
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35668) (rhsResult := 35666)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26939⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 35667 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26939⟩⟩]⟩) (none) 35666) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35669

namespace LeftMerge35670
def owner : Owner := ⟨.program ⟨257⟩, ⟨26942⟩⟩
def mergeEvent : Nat := 35670
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28018⟩⟩] } }
def rhsRaw : List Term := Proof.Events139.exact35666RawTerms
def group : MergeGroup := .relation 35668
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35668) (rhsResult := 35666)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26939⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 35667 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26939⟩⟩]⟩) (none) 35666) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28018⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28018⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35670

namespace LeftMerge35671
def owner : Owner := ⟨.program ⟨257⟩, ⟨26942⟩⟩
def mergeEvent : Nat := 35671
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13116⟩⟩, ⟨.program ⟨257⟩, ⟨26310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27463⟩⟩] } }
def rhsRaw : List Term := Proof.Events139.exact35666RawTerms
def group : MergeGroup := .relation 35668
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35668) (rhsResult := 35666)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26939⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 35667 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26939⟩⟩]⟩) (none) 35666) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13116⟩⟩, ⟨.program ⟨257⟩, ⟨26310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27463⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13116⟩⟩, ⟨.program ⟨257⟩, ⟨26310⟩⟩], [⟨.program ⟨257⟩, ⟨27463⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35671

namespace LeftMerge35672
def owner : Owner := ⟨.program ⟨257⟩, ⟨26942⟩⟩
def mergeEvent : Nat := 35672
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26480⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events139.exact35666RawTerms
def group : MergeGroup := .relation 35668
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35668) (rhsResult := 35666)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26939⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 35667 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨26939⟩⟩]⟩) (none) 35666) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26480⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26480⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35672

namespace LeftMerge35677
def owner : Owner := ⟨.program ⟨257⟩, ⟨28020⟩⟩
def mergeEvent : Nat := 35677
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13116⟩⟩, ⟨.program ⟨257⟩, ⟨26310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27463⟩⟩] } }
def leftRaw : List Term := Proof.Events139.exact35673RawTerms
def rightRaw : List Term := Proof.Events138.exact35487RawTerms
def group : MergeGroup := .operator 35673 35487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35673) (leftOrdinal := 2)
    (rightResult := 35487) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13116⟩⟩, ⟨.program ⟨257⟩, ⟨26310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27463⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13116⟩⟩, ⟨.program ⟨257⟩, ⟨26310⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27463⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨13116⟩⟩, ⟨.program ⟨257⟩, ⟨26310⟩⟩], [⟨.program ⟨257⟩, ⟨27463⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35677

namespace LeftMerge35678
def owner : Owner := ⟨.program ⟨257⟩, ⟨28020⟩⟩
def mergeEvent : Nat := 35678
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28018⟩⟩] } }
def leftRaw : List Term := Proof.Events139.exact35673RawTerms
def rightRaw : List Term := Proof.Events138.exact35487RawTerms
def group : MergeGroup := .operator 35673 35487
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35673) (leftOrdinal := 1)
    (rightResult := 35487) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28018⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28018⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7295⟩⟩, ⟨.program ⟨257⟩, ⟨9544⟩⟩, ⟨.program ⟨257⟩, ⟨28018⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35678

namespace LeftMerge35686
def owner : Owner := ⟨.program ⟨257⟩, ⟨28516⟩⟩
def mergeEvent : Nat := 35686
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28514⟩⟩] } }
def leftRaw : List Term := Proof.Events139.exact35680RawTerms
def rightRaw : List Term := Proof.Events138.exact35403RawTerms
def group : MergeGroup := .operator 35680 35403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35680) (leftOrdinal := 0)
    (rightResult := 35403) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7189⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28514⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨7189⟩⟩, ⟨.program ⟨257⟩, ⟨28514⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35686

namespace LeftMerge35687
def owner : Owner := ⟨.program ⟨257⟩, ⟨28516⟩⟩
def mergeEvent : Nat := 35687
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26480⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28514⟩⟩] } }
def leftRaw : List Term := Proof.Events139.exact35680RawTerms
def rightRaw : List Term := Proof.Events138.exact35403RawTerms
def group : MergeGroup := .operator 35680 35403
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35680) (leftOrdinal := 1)
    (rightResult := 35403) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26480⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨28514⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26480⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28514⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35687

namespace LeftMerge35689
def owner : Owner := ⟨.program ⟨257⟩, ⟨28516⟩⟩
def mergeEvent : Nat := 35689
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26480⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨27642⟩⟩] } }
def rhsRaw : List Term := Proof.Events138.exact35400RawTerms
def group : MergeGroup := .relation 35688
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 35688) (rhsResult := 35400)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26480⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨28514⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨28514⟩⟩) ⟨27642⟩ 35400) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27642⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩, ⟨.program ⟨257⟩, ⟨26480⟩⟩], [⟨.program ⟨257⟩, ⟨27642⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge35689

namespace LeftMerge35703
def owner : Owner := ⟨.program ⟨257⟩, ⟨27339⟩⟩
def mergeEvent : Nat := 35703
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27336⟩⟩] } }
def leftRaw : List Term := Proof.Events125.exact32120RawTerms
def rightRaw : List Term := Proof.Events139.exact35697RawTerms
def group : MergeGroup := .operator 32120 35697
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 32120) (leftOrdinal := 0)
    (rightResult := 35697) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨11545⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨27336⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨11545⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨27336⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35703

namespace LeftMerge35824
def owner : Owner := ⟨.program ⟨257⟩, ⟨27804⟩⟩
def mergeEvent : Nat := 35824
def frameStart : Nat := 35758
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨26480⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events139.exact35820RawTerms
def rightRaw : List Term := Proof.Events139.exact35818RawTerms
def group : MergeGroup := .operator 35820 35818
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 35820) (leftOrdinal := 0)
    (rightResult := 35818) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨26480⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨26480⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge35824

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
