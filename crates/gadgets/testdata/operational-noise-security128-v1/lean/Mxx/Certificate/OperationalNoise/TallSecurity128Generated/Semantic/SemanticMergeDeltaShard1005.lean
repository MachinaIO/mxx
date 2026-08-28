import Mxx.Certificate.OperationalNoise.CertificateSemantics
import Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Proof.History

set_option autoImplicit false
set_option relaxedAutoImplicit false

namespace Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic

open Mxx.Certificate.OperationalNoise
open CertificateABI
open CertificateSemantics
open EventReplay

namespace LeftMerge164088
def owner : Owner := ⟨.program ⟨257⟩, ⟨50130⟩⟩
def mergeEvent : Nat := 164088
def frameStart : Nat := 164009
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩] } }
def leftRaw : List Term := Proof.Events640.exact164083RawTerms
def rightRaw : List Term := Proof.Events640.exact164060RawTerms
def group : MergeGroup := .operator 164083 164060
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164083) (leftOrdinal := 1)
    (rightResult := 164060) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨50129⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164088

namespace LeftMerge164090
def owner : Owner := ⟨.program ⟨257⟩, ⟨50130⟩⟩
def mergeEvent : Nat := 164090
def frameStart : Nat := 164009
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49337⟩⟩] } }
def rhsRaw : List Term := Proof.Events640.exact164057RawTerms
def group : MergeGroup := .relation 164089
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164089) (rhsResult := 164057)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨48180⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩]⟩)
    (outerCoefficient := (-1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .gadget (⟨.program ⟨257⟩, ⟨6908⟩⟩) (⟨.program ⟨257⟩, ⟨50129⟩⟩) ⟨49337⟩ 164057) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨49337⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨48180⟩⟩], [⟨.program ⟨257⟩, ⟨49337⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164090

namespace LeftMerge164098
def owner : Owner := ⟨.program ⟨257⟩, ⟨48416⟩⟩
def mergeEvent : Nat := 164098
def frameStart : Nat := 164009
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨48415⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events640.exact164071RawTerms
def rightRaw : List Term := Proof.Events640.exact164094RawTerms
def group : MergeGroup := .operator 164071 164094
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164071) (leftOrdinal := 0)
    (rightResult := 164094) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48415⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164098

namespace LeftMerge164115
def owner : Owner := ⟨.program ⟨257⟩, ⟨48979⟩⟩
def mergeEvent : Nat := 164115
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }
def rhsRaw : List Term := Proof.Events641.exact164112RawTerms
def group : MergeGroup := .relation 164114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164114) (rhsResult := 164112)
    (sourceTermOrdinal := 1) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48976⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 164113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48976⟩⟩]⟩) (none) 164112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7232⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7232⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164115

namespace LeftMerge164116
def owner : Owner := ⟨.program ⟨257⟩, ⟨48979⟩⟩
def mergeEvent : Nat := 164116
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩] } }
def rhsRaw : List Term := Proof.Events641.exact164112RawTerms
def group : MergeGroup := .relation 164114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164114) (rhsResult := 164112)
    (sourceTermOrdinal := 0) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48976⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 164113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48976⟩⟩]⟩) (none) 164112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164116

namespace LeftMerge164117
def owner : Owner := ⟨.program ⟨257⟩, ⟨48979⟩⟩
def mergeEvent : Nat := 164117
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49337⟩⟩] } }
def rhsRaw : List Term := Proof.Events641.exact164112RawTerms
def group : MergeGroup := .relation 164114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164114) (rhsResult := 164112)
    (sourceTermOrdinal := 2) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48976⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 164113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48976⟩⟩]⟩) (none) 164112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49337⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48180⟩⟩], [⟨.program ⟨257⟩, ⟨49337⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164117

namespace LeftMerge164118
def owner : Owner := ⟨.program ⟨257⟩, ⟨48979⟩⟩
def mergeEvent : Nat := 164118
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48415⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def rhsRaw : List Term := Proof.Events641.exact164112RawTerms
def group : MergeGroup := .relation 164114
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.relation (application := 164114) (rhsResult := 164112)
    (sourceTermOrdinal := 3) (source := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48976⟩⟩]⟩)
    (outerCoefficient := (1)) (orderedStart := 0)
    (orderedEndExclusive := 2) (rule := .universal 164113 (⟨[], [⟨.program ⟨257⟩, ⟨35⟩⟩, ⟨.program ⟨257⟩, ⟨48976⟩⟩]⟩) (none) 164112) (rhsTerms := rhsRaw)
    (rhsTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨48415⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48415⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164118

namespace LeftMerge164123
def owner : Owner := ⟨.program ⟨257⟩, ⟨50132⟩⟩
def mergeEvent : Nat := 164123
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩] } }
def leftRaw : List Term := Proof.Events641.exact164119RawTerms
def rightRaw : List Term := Proof.Events640.exact163941RawTerms
def group : MergeGroup := .operator 164119 163941
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164119) (leftOrdinal := 0)
    (rightResult := 163941) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7196⟩⟩, ⟨.program ⟨257⟩, ⟨50129⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164123

namespace LeftMerge164124
def owner : Owner := ⟨.program ⟨257⟩, ⟨50132⟩⟩
def mergeEvent : Nat := 164124
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49337⟩⟩] } }
def leftRaw : List Term := Proof.Events641.exact164119RawTerms
def rightRaw : List Term := Proof.Events640.exact163941RawTerms
def group : MergeGroup := .operator 164119 163941
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164119) (leftOrdinal := 2)
    (rightResult := 163941) (rightOrdinal := 1) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49337⟩⟩] } }) (rightTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48180⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨49337⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨48180⟩⟩], [⟨.program ⟨257⟩, ⟨49337⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164124

namespace LeftMerge164150
def owner : Owner := ⟨.program ⟨257⟩, ⟨45253⟩⟩
def mergeEvent : Nat := 164150
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events029.exact7597RawTerms
def rightRaw : List Term := Proof.Events639.exact163653RawTerms
def group : MergeGroup := .operator 7597 163653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7597) (leftOrdinal := 0)
    (rightResult := 163653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164150

namespace LeftMerge164155
def owner : Owner := ⟨.program ⟨257⟩, ⟨9046⟩⟩
def mergeEvent : Nat := 164155
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163523RawTerms
def rightRaw : List Term := Proof.Events068.exact17581RawTerms
def group : MergeGroup := .operator 163523 17581
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163523) (leftOrdinal := 0)
    (rightResult := 17581) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164155

namespace LeftMerge164172
def owner : Owner := ⟨.program ⟨257⟩, ⟨45256⟩⟩
def mergeEvent : Nat := 164172
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events641.exact164166RawTerms
def rightRaw : List Term := Proof.Events029.exact7600RawTerms
def group : MergeGroup := .operator 164166 7600
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164166) (leftOrdinal := 1)
    (rightResult := 7600) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14841⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩, ⟨.program ⟨257⟩, ⟨45250⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164172

namespace LeftMerge164173
def owner : Owner := ⟨.program ⟨257⟩, ⟨45256⟩⟩
def mergeEvent : Nat := 164173
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }
def leftRaw : List Term := Proof.Events641.exact164166RawTerms
def rightRaw : List Term := Proof.Events029.exact7600RawTerms
def group : MergeGroup := .operator 164166 7600
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164166) (leftOrdinal := 0)
    (rightResult := 7600) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7284⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14841⟩⟩], orderedFactors := [] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩], [⟨.program ⟨257⟩, ⟨7284⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164173

namespace LeftMerge164178
def owner : Owner := ⟨.program ⟨257⟩, ⟨14842⟩⟩
def mergeEvent : Nat := 164178
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }
def leftRaw : List Term := Proof.Events029.exact7600RawTerms
def rightRaw : List Term := Proof.Events639.exact163653RawTerms
def group : MergeGroup := .operator 7600 163653
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 7600) (leftOrdinal := 0)
    (rightResult := 163653) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨14841⟩⟩], orderedFactors := [] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164178

namespace LeftMerge164183
def owner : Owner := ⟨.program ⟨257⟩, ⟨9063⟩⟩
def mergeEvent : Nat := 164183
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } }
def leftRaw : List Term := Proof.Events638.exact163523RawTerms
def rightRaw : List Term := Proof.Events068.exact17622RawTerms
def group : MergeGroup := .operator 163523 17622
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 163523) (leftOrdinal := 0)
    (rightResult := 17622) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨7301⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩], [⟨.program ⟨257⟩, ⟨7301⟩⟩]⟩) (signedContribution := (1)) <;> rfl
end LeftMerge164183

namespace LeftMerge164200
def owner : Owner := ⟨.program ⟨257⟩, ⟨14845⟩⟩
def mergeEvent : Nat := 164200
def frameStart : Nat := 0
def delta : ExactTerm Owner := { coefficient := (-1), key := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩] } }
def leftRaw : List Term := Proof.Events641.exact164194RawTerms
def rightRaw : List Term := Proof.Events068.exact17611RawTerms
def group : MergeGroup := .operator 164194 17611
theorem deltaAt : MergeDeltaAt history mergeEvent frameStart owner group delta := by
  unfold group delta
  apply MergeDeltaAt.operator (leftResult := 164194) (leftOrdinal := 1)
    (rightResult := 17611) (rightOrdinal := 0) (leftTerms := leftRaw)
    (rightTerms := rightRaw) (leftTerm := { coefficient := (-1), monomial := { centralFactors := [⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩], orderedFactors := [⟨.program ⟨257⟩, ⟨6908⟩⟩] } }) (rightTerm := { coefficient := (1), monomial := { centralFactors := [], orderedFactors := [⟨.program ⟨257⟩, ⟨9562⟩⟩] } })
    (output := ⟨[⟨.program ⟨257⟩, ⟨6453⟩⟩, ⟨.program ⟨257⟩, ⟨14841⟩⟩], [⟨.program ⟨257⟩, ⟨6908⟩⟩, ⟨.program ⟨257⟩, ⟨9562⟩⟩]⟩) (signedContribution := (-1)) <;> rfl
end LeftMerge164200

end Mxx.Certificate.OperationalNoise.TallSecurity128Generated.Semantic
