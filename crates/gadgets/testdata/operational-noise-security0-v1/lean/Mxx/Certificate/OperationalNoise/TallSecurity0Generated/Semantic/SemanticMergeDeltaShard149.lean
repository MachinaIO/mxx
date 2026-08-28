import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge25561
def owner : Owner := ⟨.program ⟨214⟩, ⟨28341⟩⟩
def mergeEvent : Nat := 25561
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩] } }
def leftRaw : List Term := Proof.Events099.exact25554RawTerms
def rightRaw : List Term := Proof.Events098.exact25277RawTerms
def group : MergeGroup := .operator 25554 25277
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25554) (leftOrdinal := 1)
    (rightResult := 25277) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28339⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25561

namespace LeftMerge25563
def owner : Owner := ⟨.program ⟨214⟩, ⟨28341⟩⟩
def mergeEvent : Nat := 25563
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24297⟩⟩] } }
def rhsRaw : List Term := Proof.Events098.exact25274RawTerms
def group : MergeGroup := .relation 25562
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25562) (rhsResult := 25274)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28339⟩⟩) ⟨24297⟩ 25274) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24297⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16190⟩⟩], [⟨.program ⟨214⟩, ⟨24297⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25563

namespace LeftMerge25577
def owner : Owner := ⟨.program ⟨214⟩, ⟨21703⟩⟩
def mergeEvent : Nat := 25577
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21700⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events099.exact25571RawTerms
def group : MergeGroup := .operator 21512 25571
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 25571) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨21700⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21700⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25577

namespace LeftMerge25698
def owner : Owner := ⟨.program ⟨214⟩, ⟨16232⟩⟩
def mergeEvent : Nat := 25698
def frameStart : Nat := 25632
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events100.exact25694RawTerms
def rightRaw : List Term := Proof.Events100.exact25692RawTerms
def group : MergeGroup := .operator 25694 25692
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25694) (leftOrdinal := 0)
    (rightResult := 25692) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25698

namespace LeftMerge25710
def owner : Owner := ⟨.program ⟨214⟩, ⟨28340⟩⟩
def mergeEvent : Nat := 25710
def frameStart : Nat := 25632
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩] } }
def leftRaw : List Term := Proof.Events100.exact25706RawTerms
def rightRaw : List Term := Proof.Events100.exact25683RawTerms
def group : MergeGroup := .operator 25706 25683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25706) (leftOrdinal := 0)
    (rightResult := 25683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28339⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25710

namespace LeftMerge25711
def owner : Owner := ⟨.program ⟨214⟩, ⟨28340⟩⟩
def mergeEvent : Nat := 25711
def frameStart : Nat := 25632
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩] } }
def leftRaw : List Term := Proof.Events100.exact25706RawTerms
def rightRaw : List Term := Proof.Events100.exact25683RawTerms
def group : MergeGroup := .operator 25706 25683
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25706) (leftOrdinal := 1)
    (rightResult := 25683) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨28339⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨16190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25711

namespace LeftMerge25713
def owner : Owner := ⟨.program ⟨214⟩, ⟨28340⟩⟩
def mergeEvent : Nat := 25713
def frameStart : Nat := 25632
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24297⟩⟩] } }
def rhsRaw : List Term := Proof.Events100.exact25680RawTerms
def group : MergeGroup := .relation 25712
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25712) (rhsResult := 25680)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨16190⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨28339⟩⟩) ⟨24297⟩ 25680) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24297⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨16190⟩⟩], [⟨.program ⟨214⟩, ⟨24297⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25713

namespace LeftMerge25721
def owner : Owner := ⟨.program ⟨214⟩, ⟨18390⟩⟩
def mergeEvent : Nat := 25721
def frameStart : Nat := 25632
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨18379⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events100.exact25694RawTerms
def rightRaw : List Term := Proof.Events100.exact25717RawTerms
def group : MergeGroup := .operator 25694 25717
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25694) (leftOrdinal := 0)
    (rightResult := 25717) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18379⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨18379⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25721

namespace LeftMerge25738
def owner : Owner := ⟨.program ⟨214⟩, ⟨21703⟩⟩
def mergeEvent : Nat := 25738
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }
def rhsRaw : List Term := Proof.Events100.exact25735RawTerms
def group : MergeGroup := .relation 25737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25737) (rhsResult := 25735)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21700⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25736 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21700⟩⟩]⟩) (none) 25735) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6727⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6727⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25738

namespace LeftMerge25739
def owner : Owner := ⟨.program ⟨214⟩, ⟨21703⟩⟩
def mergeEvent : Nat := 25739
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩] } }
def rhsRaw : List Term := Proof.Events100.exact25735RawTerms
def group : MergeGroup := .relation 25737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25737) (rhsResult := 25735)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21700⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25736 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21700⟩⟩]⟩) (none) 25735) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25739

namespace LeftMerge25740
def owner : Owner := ⟨.program ⟨214⟩, ⟨21703⟩⟩
def mergeEvent : Nat := 25740
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24297⟩⟩] } }
def rhsRaw : List Term := Proof.Events100.exact25735RawTerms
def group : MergeGroup := .relation 25737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25737) (rhsResult := 25735)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21700⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25736 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21700⟩⟩]⟩) (none) 25735) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24297⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16190⟩⟩], [⟨.program ⟨214⟩, ⟨24297⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25740

namespace LeftMerge25741
def owner : Owner := ⟨.program ⟨214⟩, ⟨21703⟩⟩
def mergeEvent : Nat := 25741
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18379⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def rhsRaw : List Term := Proof.Events100.exact25735RawTerms
def group : MergeGroup := .relation 25737
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 25737) (rhsResult := 25735)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21700⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 25736 (⟨[], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨21700⟩⟩]⟩) (none) 25735) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨18379⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨18379⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25741

namespace LeftMerge25746
def owner : Owner := ⟨.program ⟨214⟩, ⟨28342⟩⟩
def mergeEvent : Nat := 25746
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩] } }
def leftRaw : List Term := Proof.Events100.exact25742RawTerms
def rightRaw : List Term := Proof.Events099.exact25564RawTerms
def group : MergeGroup := .operator 25742 25564
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25742) (leftOrdinal := 0)
    (rightResult := 25564) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6699⟩⟩, ⟨.program ⟨214⟩, ⟨28339⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25746

namespace LeftMerge25747
def owner : Owner := ⟨.program ⟨214⟩, ⟨28342⟩⟩
def mergeEvent : Nat := 25747
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24297⟩⟩] } }
def leftRaw : List Term := Proof.Events100.exact25742RawTerms
def rightRaw : List Term := Proof.Events099.exact25564RawTerms
def group : MergeGroup := .operator 25742 25564
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 25742) (leftOrdinal := 2)
    (rightResult := 25564) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24297⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16190⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨24297⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨16190⟩⟩], [⟨.program ⟨214⟩, ⟨24297⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge25747

namespace LeftMerge25773
def owner : Owner := ⟨.program ⟨214⟩, ⟨11566⟩⟩
def mergeEvent : Nat := 25773
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11565⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events004.exact1049RawTerms
def rightRaw : List Term := Proof.Events083.exact21420RawTerms
def group : MergeGroup := .operator 1049 21420
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1049) (leftOrdinal := 0)
    (rightResult := 21420) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨11565⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨11565⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25773

namespace LeftMerge25778
def owner : Owner := ⟨.program ⟨214⟩, ⟨7350⟩⟩
def mergeEvent : Nat := 25778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21290RawTerms
def rightRaw : List Term := Proof.Events042.exact10981RawTerms
def group : MergeGroup := .operator 21290 10981
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21290) (leftOrdinal := 0)
    (rightResult := 10981) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6780⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6780⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge25778

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
