import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge210597
def owner : Owner := ⟨.program ⟨257⟩, ⟨28775⟩⟩
def mergeEvent : Nat := 210597
def frameStart : Nat := 210567
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], orderedFactors := [] } }
def leftRaw : List Term := Proof.Events822.exact210593RawTerms
def rightRaw : List Term := Proof.Events822.exact210590RawTerms
def group : MergeGroup := .operator 210593 210590
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210593) (leftOrdinal := 0)
    (rightResult := 210590) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13281⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨28774⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], []⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210597

namespace LeftMerge210627
def owner : Owner := ⟨.program ⟨257⟩, ⟨30368⟩⟩
def mergeEvent : Nat := 210627
def frameStart : Nat := 210567
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events822.exact210623RawTerms
def rightRaw : List Term := Proof.Events822.exact210621RawTerms
def group : MergeGroup := .operator 210623 210621
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210623) (leftOrdinal := 0)
    (rightResult := 210621) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210627

namespace LeftMerge210650
def owner : Owner := ⟨.program ⟨257⟩, ⟨9549⟩⟩
def mergeEvent : Nat := 210650
def frameStart : Nat := 210567
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }
def leftRaw : List Term := Proof.Events822.exact210646RawTerms
def rightRaw : List Term := Proof.Events822.exact210643RawTerms
def group : MergeGroup := .operator 210646 210643
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210646) (leftOrdinal := 0)
    (rightResult := 210643) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9547⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210650

namespace LeftMerge210659
def owner : Owner := ⟨.program ⟨257⟩, ⟨30602⟩⟩
def mergeEvent : Nat := 210659
def frameStart : Nat := 210567
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30599⟩⟩] } }
def leftRaw : List Term := Proof.Events822.exact210655RawTerms
def rightRaw : List Term := Proof.Events822.exact210612RawTerms
def group : MergeGroup := .operator 210655 210612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210655) (leftOrdinal := 0)
    (rightResult := 210612) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30599⟩⟩] } })
    (output := ⟨[], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30599⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210659

namespace LeftMerge210660
def owner : Owner := ⟨.program ⟨257⟩, ⟨30602⟩⟩
def mergeEvent : Nat := 210660
def frameStart : Nat := 210567
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30599⟩⟩] } }
def leftRaw : List Term := Proof.Events822.exact210655RawTerms
def rightRaw : List Term := Proof.Events822.exact210612RawTerms
def group : MergeGroup := .operator 210655 210612
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210655) (leftOrdinal := 1)
    (rightResult := 210612) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30599⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30599⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210660

namespace LeftMerge210662
def owner : Owner := ⟨.program ⟨257⟩, ⟨30602⟩⟩
def mergeEvent : Nat := 210662
def frameStart : Nat := 210567
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30089⟩⟩] } }
def rhsRaw : List Term := Proof.Events822.exact210609RawTerms
def group : MergeGroup := .relation 210661
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 210661) (rhsResult := 210609)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30599⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30599⟩⟩) ⟨30089⟩ 210609) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30089⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], [⟨.program ⟨257⟩, ⟨30089⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210662

namespace LeftMerge210670
def owner : Owner := ⟨.program ⟨257⟩, ⟨29090⟩⟩
def mergeEvent : Nat := 210670
def frameStart : Nat := 210567
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨29088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events822.exact210623RawTerms
def rightRaw : List Term := Proof.Events822.exact210666RawTerms
def group : MergeGroup := .operator 210623 210666
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210623) (leftOrdinal := 0)
    (rightResult := 210666) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29088⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨29088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210670

namespace LeftMerge210687
def owner : Owner := ⟨.program ⟨257⟩, ⟨29532⟩⟩
def mergeEvent : Nat := 210687
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }
def rhsRaw : List Term := Proof.Events822.exact210684RawTerms
def group : MergeGroup := .relation 210686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 210686) (rhsResult := 210684)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29529⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 210685 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29529⟩⟩]⟩) (none) 210684) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210687

namespace LeftMerge210688
def owner : Owner := ⟨.program ⟨257⟩, ⟨29532⟩⟩
def mergeEvent : Nat := 210688
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30599⟩⟩] } }
def rhsRaw : List Term := Proof.Events822.exact210684RawTerms
def group : MergeGroup := .relation 210686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 210686) (rhsResult := 210684)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29529⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 210685 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29529⟩⟩]⟩) (none) 210684) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30599⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30599⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210688

namespace LeftMerge210689
def owner : Owner := ⟨.program ⟨257⟩, ⟨29532⟩⟩
def mergeEvent : Nat := 210689
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30089⟩⟩] } }
def rhsRaw : List Term := Proof.Events822.exact210684RawTerms
def group : MergeGroup := .relation 210686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 210686) (rhsResult := 210684)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29529⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 210685 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29529⟩⟩]⟩) (none) 210684) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30089⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], [⟨.program ⟨257⟩, ⟨30089⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210689

namespace LeftMerge210690
def owner : Owner := ⟨.program ⟨257⟩, ⟨29532⟩⟩
def mergeEvent : Nat := 210690
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events822.exact210684RawTerms
def group : MergeGroup := .relation 210686
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 210686) (rhsResult := 210684)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29529⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 210685 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨29529⟩⟩]⟩) (none) 210684) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨29088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210690

namespace LeftMerge210695
def owner : Owner := ⟨.program ⟨257⟩, ⟨30601⟩⟩
def mergeEvent : Nat := 210695
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30089⟩⟩] } }
def leftRaw : List Term := Proof.Events823.exact210691RawTerms
def rightRaw : List Term := Proof.Events822.exact210505RawTerms
def group : MergeGroup := .operator 210691 210505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210691) (leftOrdinal := 2)
    (rightResult := 210505) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30089⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30089⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨13281⟩⟩, ⟨.program ⟨257⟩, ⟨28774⟩⟩], [⟨.program ⟨257⟩, ⟨30089⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210695

namespace LeftMerge210696
def owner : Owner := ⟨.program ⟨257⟩, ⟨30601⟩⟩
def mergeEvent : Nat := 210696
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30599⟩⟩] } }
def leftRaw : List Term := Proof.Events823.exact210691RawTerms
def rightRaw : List Term := Proof.Events822.exact210505RawTerms
def group : MergeGroup := .operator 210691 210505
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210691) (leftOrdinal := 1)
    (rightResult := 210505) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30599⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30599⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩, ⟨.program ⟨257⟩, ⟨30599⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210696

namespace LeftMerge210704
def owner : Owner := ⟨.program ⟨257⟩, ⟨30971⟩⟩
def mergeEvent : Nat := 210704
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30969⟩⟩] } }
def leftRaw : List Term := Proof.Events823.exact210698RawTerms
def rightRaw : List Term := Proof.Events821.exact210421RawTerms
def group : MergeGroup := .operator 210698 210421
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210698) (leftOrdinal := 0)
    (rightResult := 210421) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7190⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30969⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩], [⟨.program ⟨257⟩, ⟨7190⟩⟩, ⟨.program ⟨257⟩, ⟨30969⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge210704

namespace LeftMerge210705
def owner : Owner := ⟨.program ⟨257⟩, ⟨30971⟩⟩
def mergeEvent : Nat := 210705
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30969⟩⟩] } }
def leftRaw : List Term := Proof.Events823.exact210698RawTerms
def rightRaw : List Term := Proof.Events821.exact210421RawTerms
def group : MergeGroup := .operator 210698 210421
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 210698) (leftOrdinal := 1)
    (rightResult := 210421) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30969⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30969⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210705

namespace LeftMerge210707
def owner : Owner := ⟨.program ⟨257⟩, ⟨30971⟩⟩
def mergeEvent : Nat := 210707
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29088⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨30241⟩⟩] } }
def rhsRaw : List Term := Proof.Events821.exact210418RawTerms
def group : MergeGroup := .relation 210706
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 210706) (rhsResult := 210418)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29088⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30969⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨30969⟩⟩) ⟨30241⟩ 210418) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30241⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨5896⟩⟩, ⟨.program ⟨257⟩, ⟨29088⟩⟩], [⟨.program ⟨257⟩, ⟨30241⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge210707

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
