import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge29651
def owner : Owner := ⟨.program ⟨214⟩, ⟨10510⟩⟩
def mergeEvent : Nat := 29651
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events115.exact29645RawTerms
def rightRaw : List Term := Proof.Events004.exact1236RawTerms
def group : MergeGroup := .operator 29645 1236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29645) (leftOrdinal := 1)
    (rightResult := 1236) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9415⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29651

namespace LeftMerge29652
def owner : Owner := ⟨.program ⟨214⟩, ⟨10510⟩⟩
def mergeEvent : Nat := 29652
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }
def leftRaw : List Term := Proof.Events115.exact29645RawTerms
def rightRaw : List Term := Proof.Events004.exact1236RawTerms
def group : MergeGroup := .operator 29645 1236
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29645) (leftOrdinal := 0)
    (rightResult := 1236) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9415⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29652

namespace LeftMerge29657
def owner : Owner := ⟨.program ⟨214⟩, ⟨9416⟩⟩
def mergeEvent : Nat := 29657
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events004.exact1236RawTerms
def rightRaw : List Term := Proof.Events083.exact21420RawTerms
def group : MergeGroup := .operator 1236 21420
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 1236) (leftOrdinal := 0)
    (rightResult := 21420) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9415⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29657

namespace LeftMerge29662
def owner : Owner := ⟨.program ⟨214⟩, ⟨7341⟩⟩
def mergeEvent : Nat := 29662
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩] } }
def leftRaw : List Term := Proof.Events083.exact21290RawTerms
def rightRaw : List Term := Proof.Events058.exact15030RawTerms
def group : MergeGroup := .operator 21290 15030
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21290) (leftOrdinal := 0)
    (rightResult := 15030) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29662

namespace LeftMerge29679
def owner : Owner := ⟨.program ⟨214⟩, ⟨9419⟩⟩
def mergeEvent : Nat := 29679
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }
def leftRaw : List Term := Proof.Events115.exact29673RawTerms
def rightRaw : List Term := Proof.Events058.exact15019RawTerms
def group : MergeGroup := .operator 29673 15019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29673) (leftOrdinal := 1)
    (rightResult := 15019) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7831⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29679

namespace LeftMerge29681
def owner : Owner := ⟨.program ⟨214⟩, ⟨9419⟩⟩
def mergeEvent : Nat := 29681
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }
def rhsRaw : List Term := Proof.Events058.exact14989RawTerms
def group : MergeGroup := .relation 29680
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29680) (rhsResult := 14989)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨7831⟩⟩) ⟨6772⟩ 14989) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29681

namespace LeftMerge29682
def owner : Owner := ⟨.program ⟨214⟩, ⟨9419⟩⟩
def mergeEvent : Nat := 29682
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }
def leftRaw : List Term := Proof.Events115.exact29673RawTerms
def rightRaw : List Term := Proof.Events058.exact15019RawTerms
def group : MergeGroup := .operator 29673 15019
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29673) (leftOrdinal := 0)
    (rightResult := 15019) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7831⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29682

namespace LeftMerge29687
def owner : Owner := ⟨.program ⟨214⟩, ⟨10511⟩⟩
def mergeEvent : Nat := 29687
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }
def leftRaw : List Term := Proof.Events115.exact29683RawTerms
def rightRaw : List Term := Proof.Events115.exact29653RawTerms
def group : MergeGroup := .operator 29683 29653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29683) (leftOrdinal := 1)
    (rightResult := 29653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6772⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩], [⟨.program ⟨214⟩, ⟨6772⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29687

namespace LeftMerge29695
def owner : Owner := ⟨.program ⟨214⟩, ⟨24927⟩⟩
def mergeEvent : Nat := 29695
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24926⟩⟩] } }
def leftRaw : List Term := Proof.Events115.exact29689RawTerms
def rightRaw : List Term := Proof.Events115.exact29625RawTerms
def group : MergeGroup := .operator 29689 29625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29689) (leftOrdinal := 1)
    (rightResult := 29625) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24926⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24926⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29695

namespace LeftMerge29697
def owner : Owner := ⟨.program ⟨214⟩, ⟨24927⟩⟩
def mergeEvent : Nat := 29697
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨22960⟩⟩] } }
def rhsRaw : List Term := Proof.Events115.exact29622RawTerms
def group : MergeGroup := .relation 29696
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 29696) (rhsResult := 29622)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩, ⟨.program ⟨214⟩, ⟨24926⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨214⟩, ⟨6544⟩⟩) (⟨.program ⟨214⟩, ⟨24926⟩⟩) ⟨22960⟩ 29622) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨22960⟩⟩] } }) (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩, ⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], [⟨.program ⟨214⟩, ⟨22960⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge29697

namespace LeftMerge29698
def owner : Owner := ⟨.program ⟨214⟩, ⟨24927⟩⟩
def mergeEvent : Nat := 29698
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24926⟩⟩] } }
def leftRaw : List Term := Proof.Events115.exact29689RawTerms
def rightRaw : List Term := Proof.Events115.exact29625RawTerms
def group : MergeGroup := .operator 29689 29625
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29689) (leftOrdinal := 0)
    (rightResult := 29625) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24926⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24926⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29698

namespace LeftMerge29712
def owner : Owner := ⟨.program ⟨214⟩, ⟨19039⟩⟩
def mergeEvent : Nat := 29712
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19036⟩⟩] } }
def leftRaw : List Term := Proof.Events084.exact21512RawTerms
def rightRaw : List Term := Proof.Events116.exact29706RawTerms
def group : MergeGroup := .operator 21512 29706
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 21512) (leftOrdinal := 0)
    (rightResult := 29706) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨5517⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨19036⟩⟩] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨5517⟩⟩], [⟨.program ⟨214⟩, ⟨6⟩⟩, ⟨.program ⟨214⟩, ⟨19036⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29712

namespace LeftMerge29791
def owner : Owner := ⟨.program ⟨214⟩, ⟨10505⟩⟩
def mergeEvent : Nat := 29791
def frameStart : Nat := 29761
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events116.exact29787RawTerms
def rightRaw : List Term := Proof.Events116.exact29784RawTerms
def group : MergeGroup := .operator 29787 29784
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29787) (leftOrdinal := 0)
    (rightResult := 29784) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9415⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨10504⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29791

namespace LeftMerge29821
def owner : Owner := ⟨.program ⟨214⟩, ⟨10590⟩⟩
def mergeEvent : Nat := 29821
def frameStart : Nat := 29761
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }
def leftRaw : List Term := Proof.Events116.exact29817RawTerms
def rightRaw : List Term := Proof.Events116.exact29815RawTerms
def group : MergeGroup := .operator 29817 29815
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29817) (leftOrdinal := 0)
    (rightResult := 29815) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6544⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨214⟩, ⟨9415⟩⟩, ⟨.program ⟨214⟩, ⟨10504⟩⟩], [⟨.program ⟨214⟩, ⟨6544⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29821

namespace LeftMerge29844
def owner : Owner := ⟨.program ⟨214⟩, ⟨7833⟩⟩
def mergeEvent : Nat := 29844
def frameStart : Nat := 29761
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }
def leftRaw : List Term := Proof.Events116.exact29840RawTerms
def rightRaw : List Term := Proof.Events116.exact29837RawTerms
def group : MergeGroup := .operator 29840 29837
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29840) (leftOrdinal := 0)
    (rightResult := 29837) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨7831⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29844

namespace LeftMerge29853
def owner : Owner := ⟨.program ⟨214⟩, ⟨24929⟩⟩
def mergeEvent : Nat := 29853
def frameStart : Nat := 29761
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24926⟩⟩] } }
def leftRaw : List Term := Proof.Events116.exact29849RawTerms
def rightRaw : List Term := Proof.Events116.exact29806RawTerms
def group : MergeGroup := .operator 29849 29806
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 29849) (leftOrdinal := 0)
    (rightResult := 29806) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨214⟩, ⟨24926⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨214⟩, ⟨6771⟩⟩, ⟨.program ⟨214⟩, ⟨7831⟩⟩, ⟨.program ⟨214⟩, ⟨24926⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge29853

end Mxx.Certificate.OperationalNoise.TallSecurity0Generated.Semantic
