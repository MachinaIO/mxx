import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge195776
def owner : Owner := ⟨.program ⟨257⟩, ⟨35539⟩⟩
def mergeEvent : Nat := 195776
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩] } }
def rhsRaw : List Term := Proof.Events764.exact195772RawTerms
def group : MergeGroup := .relation 195774
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195774) (rhsResult := 195772)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35536⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 195773 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35536⟩⟩]⟩) (none) 195772) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195776

namespace LeftMerge195777
def owner : Owner := ⟨.program ⟨257⟩, ⟨35539⟩⟩
def mergeEvent : Nat := 195777
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35919⟩⟩] } }
def rhsRaw : List Term := Proof.Events764.exact195772RawTerms
def group : MergeGroup := .relation 195774
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195774) (rhsResult := 195772)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35536⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 195773 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35536⟩⟩]⟩) (none) 195772) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35919⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨35919⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195777

namespace LeftMerge195778
def owner : Owner := ⟨.program ⟨257⟩, ⟨35539⟩⟩
def mergeEvent : Nat := 195778
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34989⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events764.exact195772RawTerms
def group : MergeGroup := .relation 195774
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195774) (rhsResult := 195772)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35536⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 195773 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨35536⟩⟩]⟩) (none) 195772) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨34989⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34989⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195778

namespace LeftMerge195783
def owner : Owner := ⟨.program ⟨257⟩, ⟨36682⟩⟩
def mergeEvent : Nat := 195783
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩] } }
def leftRaw : List Term := Proof.Events764.exact195779RawTerms
def rightRaw : List Term := Proof.Events764.exact195601RawTerms
def group : MergeGroup := .operator 195779 195601
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195779) (leftOrdinal := 0)
    (rightResult := 195601) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7191⟩⟩, ⟨.program ⟨257⟩, ⟨36679⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195783

namespace LeftMerge195784
def owner : Owner := ⟨.program ⟨257⟩, ⟨36682⟩⟩
def mergeEvent : Nat := 195784
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35919⟩⟩] } }
def leftRaw : List Term := Proof.Events764.exact195779RawTerms
def rightRaw : List Term := Proof.Events764.exact195601RawTerms
def group : MergeGroup := .operator 195779 195601
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195779) (leftOrdinal := 2)
    (rightResult := 195601) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35919⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨35919⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨34764⟩⟩], [⟨.program ⟨257⟩, ⟨35919⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195784

namespace LeftMerge195810
def owner : Owner := ⟨.program ⟨257⟩, ⟨28825⟩⟩
def mergeEvent : Nat := 195810
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨28822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events035.exact9208RawTerms
def rightRaw : List Term := Proof.Events753.exact192903RawTerms
def group : MergeGroup := .operator 9208 192903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9208) (leftOrdinal := 0)
    (rightResult := 192903) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨28822⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨28822⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195810

namespace LeftMerge195815
def owner : Owner := ⟨.program ⟨257⟩, ⟨8813⟩⟩
def mergeEvent : Nat := 195815
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192773RawTerms
def rightRaw : List Term := Proof.Events078.exact20086RawTerms
def group : MergeGroup := .operator 192773 20086
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192773) (leftOrdinal := 0)
    (rightResult := 20086) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195815

namespace LeftMerge195832
def owner : Owner := ⟨.program ⟨257⟩, ⟨28828⟩⟩
def mergeEvent : Nat := 195832
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩, ⟨.program ⟨257⟩, ⟨28822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events764.exact195826RawTerms
def rightRaw : List Term := Proof.Events035.exact9211RawTerms
def group : MergeGroup := .operator 195826 9211
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195826) (leftOrdinal := 1)
    (rightResult := 9211) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨28822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13311⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩, ⟨.program ⟨257⟩, ⟨28822⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195832

namespace LeftMerge195833
def owner : Owner := ⟨.program ⟨257⟩, ⟨28828⟩⟩
def mergeEvent : Nat := 195833
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }
def leftRaw : List Term := Proof.Events764.exact195826RawTerms
def rightRaw : List Term := Proof.Events035.exact9211RawTerms
def group : MergeGroup := .operator 195826 9211
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195826) (leftOrdinal := 0)
    (rightResult := 9211) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13311⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195833

namespace LeftMerge195838
def owner : Owner := ⟨.program ⟨257⟩, ⟨13312⟩⟩
def mergeEvent : Nat := 195838
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events035.exact9211RawTerms
def rightRaw : List Term := Proof.Events753.exact192903RawTerms
def group : MergeGroup := .operator 9211 192903
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 9211) (leftOrdinal := 0)
    (rightResult := 192903) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨13311⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195838

namespace LeftMerge195843
def owner : Owner := ⟨.program ⟨257⟩, ⟨8830⟩⟩
def mergeEvent : Nat := 195843
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }
def leftRaw : List Term := Proof.Events753.exact192773RawTerms
def rightRaw : List Term := Proof.Events078.exact20127RawTerms
def group : MergeGroup := .operator 192773 20127
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 192773) (leftOrdinal := 0)
    (rightResult := 20127) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195843

namespace LeftMerge195860
def owner : Owner := ⟨.program ⟨257⟩, ⟨13315⟩⟩
def mergeEvent : Nat := 195860
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }
def leftRaw : List Term := Proof.Events765.exact195854RawTerms
def rightRaw : List Term := Proof.Events078.exact20116RawTerms
def group : MergeGroup := .operator 195854 20116
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195854) (leftOrdinal := 1)
    (rightResult := 20116) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9547⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195860

namespace LeftMerge195862
def owner : Owner := ⟨.program ⟨257⟩, ⟨13315⟩⟩
def mergeEvent : Nat := 195862
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }
def rhsRaw : List Term := Proof.Events078.exact20086RawTerms
def group : MergeGroup := .relation 195861
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 195861) (rhsResult := 20086)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨9547⟩⟩) ⟨7279⟩ 20086) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195862

namespace LeftMerge195863
def owner : Owner := ⟨.program ⟨257⟩, ⟨13315⟩⟩
def mergeEvent : Nat := 195863
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩] } }
def leftRaw : List Term := Proof.Events765.exact195854RawTerms
def rightRaw : List Term := Proof.Events078.exact20116RawTerms
def group : MergeGroup := .operator 195854 20116
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195854) (leftOrdinal := 0)
    (rightResult := 20116) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7296⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9547⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩], [⟨.program ⟨257⟩, ⟨7296⟩⟩, ⟨.program ⟨257⟩, ⟨9547⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195863

namespace LeftMerge195868
def owner : Owner := ⟨.program ⟨257⟩, ⟨28829⟩⟩
def mergeEvent : Nat := 195868
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }
def leftRaw : List Term := Proof.Events765.exact195864RawTerms
def rightRaw : List Term := Proof.Events764.exact195834RawTerms
def group : MergeGroup := .operator 195864 195834
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195864) (leftOrdinal := 1)
    (rightResult := 195834) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7279⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩], [⟨.program ⟨257⟩, ⟨7279⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge195868

namespace LeftMerge195876
def owner : Owner := ⟨.program ⟨257⟩, ⟨30622⟩⟩
def mergeEvent : Nat := 195876
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩, ⟨.program ⟨257⟩, ⟨28822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30621⟩⟩] } }
def leftRaw : List Term := Proof.Events765.exact195870RawTerms
def rightRaw : List Term := Proof.Events764.exact195806RawTerms
def group : MergeGroup := .operator 195870 195806
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 195870) (leftOrdinal := 1)
    (rightResult := 195806) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩, ⟨.program ⟨257⟩, ⟨28822⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨30621⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6173⟩⟩, ⟨.program ⟨257⟩, ⟨13311⟩⟩, ⟨.program ⟨257⟩, ⟨28822⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨30621⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge195876

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
